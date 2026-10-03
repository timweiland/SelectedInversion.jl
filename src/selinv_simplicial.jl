using SparseArrays

export selinv_simplicial

@inline function LL_col_to_LDL!(L::SparseMatrixCSC, j)
    nzval = L.nzval
    @inbounds begin
        k = L.colptr[j]
        inv_L_jj = 1 / nzval[k]
        for i in (k + 1):(L.colptr[j + 1] - 1)
            nzval[i] *= inv_L_jj
        end
        nzval[k] = inv_L_jj^2
    end
    return
end

# Index of the first entry >= x in the sorted range rowval[lo:hi] (hi + 1 if
# there is none), given rowval[lo] < x. Exponential search keeps short skips
# cheap, while skipping far ahead in a long column costs O(log distance).
@inline function _gallop(rowval, lo, hi, x)
    step = 1
    @inbounds while lo + step <= hi && rowval[lo + step] < x
        lo += step
        step <<= 1
    end
    return searchsortedfirst(rowval, x, lo + 1, min(lo + step, hi), Base.Order.Forward)
end

# Column k of Z holds the scaled factor entries Zk = L[rows, k] / L[k, k] below
# the diagonal. With S = Z[rows, rows] (already computed), set
#   Y = S * Zk,   Z[k, k] += Zk' * Y,   Zk = -Y.
# S is read from the lower triangle: for each row r_p, column r_p holds S[r_q, r_p]
# for q > p. Both row lists are sorted, so a merge finds those entries, and each
# match contributes to Y[p] and Y[q] by symmetry. Rows missing from column r_p are
# structural zeros; for a Cholesky pattern there are none.
@inline function update_col!(Y_buf, Z::SparseMatrixCSC, k)
    colptr, rowval, nzval = Z.colptr, Z.rowval, Z.nzval
    ks = colptr[k]
    m = colptr[k + 1] - ks - 1 # Ignore diagonal entry
    @inbounds begin
        for p in 1:m
            Y_buf[p] = zero(eltype(Y_buf))
        end
        for p in 1:m
            c = rowval[ks + p]
            zp = nzval[ks + p]
            cs = colptr[c]
            ce = colptr[c + 1] - 1
            yp = Y_buf[p] + nzval[cs] * zp
            a = cs + 1
            q = p + 1
            while q <= m && a <= ce
                rq = rowval[ks + q]
                ra = rowval[a]
                if ra == rq
                    v = nzval[a]
                    yp += v * nzval[ks + q]
                    Y_buf[q] += v * zp
                    a += 1
                    q += 1
                elseif ra < rq
                    a = _gallop(rowval, a, ce, rq)
                else
                    q += 1
                end
            end
            Y_buf[p] = yp
        end
        d = zero(eltype(Y_buf))
        for p in 1:m
            d += Y_buf[p] * nzval[ks + p]
            nzval[ks + p] = -Y_buf[p]
        end
        nzval[ks] += d
    end
    return
end

# Copy a CHOLMOD factor into a SparseMatrixCSC. A simplicial factor is read
# directly from its column arrays: columns may be unpacked (slack after column
# j's `nz[j]` entries), but each is sorted with the diagonal first. For an LDL
# factor, the diagonal holds D and the unit diagonal of L is implicit.
function _simplicial_factor_csc(F::SparseArrays.CHOLMOD.Factor{Tv, Ti}) where {Tv <: Real, Ti}
    s = unsafe_load(pointer(F))
    if s.xtype == SparseArrays.CHOLMOD.CHOLMOD_PATTERN
        throw(SparseArrays.CHOLMOD.CHOLMODException("only numeric factors are supported"))
    end
    Bool(s.is_super) && return sparse(F.L) # Supernodal factors are always LL
    return GC.@preserve F begin
        n = Int(s.n)
        Lp = unsafe_wrap(Array, Ptr{Ti}(s.p), n + 1)
        Lnz = unsafe_wrap(Array, Ptr{Ti}(s.nz), n)
        Li = unsafe_wrap(Array, Ptr{Ti}(s.i), Int(s.nzmax))
        Lx = unsafe_wrap(Array, Ptr{Tv}(s.x), Int(s.nzmax))
        colptr = Vector{Int}(undef, n + 1)
        colptr[1] = 1
        @inbounds for j in 1:n
            colptr[j + 1] = colptr[j] + Lnz[j]
        end
        rowval = Vector{Int}(undef, colptr[n + 1] - 1)
        nzval = Vector{Tv}(undef, colptr[n + 1] - 1)
        @inbounds for j in 1:n
            src = Lp[j]
            dst = colptr[j] - 1
            for t in 1:Lnz[j]
                rowval[dst + t] = Li[src + t] + 1
                nzval[dst + t] = Lx[src + t]
            end
        end
        SparseMatrixCSC(n, n, colptr, rowval, nzval)
    end
end

function selinv_simplicial(F::SparseArrays.CHOLMOD.Factor; depermute = false)
    Z = _simplicial_factor_csc(F)
    if Bool(unsafe_load(pointer(F)).is_ll)
        _selinv_simplicial_Z!(Z; from_ll = true)
    else
        # The sweep expects D⁻¹ on the diagonal
        for k in axes(Z, 2)
            Z.nzval[Z.colptr[k]] = inv(Z.nzval[Z.colptr[k]])
        end
        _selinv_simplicial_Z!(Z)
    end

    p = F.p
    if depermute
        # Symmetric(Z)[invperm(p), invperm(p)], but much faster
        # ... at the cost of more memory
        rows, cols, vals = findnz(sparse(Symmetric(Z, :L)))
        p_rows, p_cols = p[rows], p[cols]
        new_rows = [p_rows; p_cols]
        new_cols = [p_cols; p_rows]
        new_vals = [vals; vals]
        n = size(Z, 1)
        Z = sparse(new_rows, new_cols, new_vals, n, n, (x, y) -> x)
        return (Z = Z, p = p)
    else
        return (Z = Symmetric(Z, :L), p = p)
    end
end

# If `from_ll`, Z initially holds the LL factor and each column is converted
# to LDL form right before it is processed.
function _selinv_simplicial_Z!(Z::SparseMatrixCSC; from_ll::Bool = false)
    N = size(Z, 2)
    N == 0 && return
    max_col = maximum(j -> Z.colptr[j + 1] - Z.colptr[j], 1:N)
    Y_buf = zeros(eltype(Z), max_col - 1)
    from_ll && LL_col_to_LDL!(Z, N)
    for j in (N - 1):-1:1
        from_ll && LL_col_to_LDL!(Z, j)
        update_col!(Y_buf, Z, j)
    end
    return
end
