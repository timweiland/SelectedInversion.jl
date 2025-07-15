using SparseArrays

export selinv_simplicial

function LL_simplicial_to_LDL!(L::SparseMatrixCSC)
    for j = 1:size(L, 2)
        L_j = @view(L.nzval[nzrange(L, j)])
        L_jj = L_j[1]
        L_j ./= L_jj
        L_j[1] = (1 / L_jj^2)
    end
end

function update_col!(Y_buf, Z, k)
    nzrng = nzrange(Z, k)[2:end]
    nnz = length(nzrng) # Ignore diagonal entry
    row_rng = @view(Z.rowval[nzrng])
    Zk = @view(Z.nzval[nzrng])
    Y_buf_local = @view(Y_buf[1:nnz])
    fill!(Y_buf_local, 0.0)
    @inbounds for j = 1:nnz
        j_row = row_rng[j]
        Y_buf_local[j] += Z[j_row, j_row] * Zk[j]
        @inbounds for i = (j+1):nnz
            i_row = row_rng[i]
            Y_buf_local[i] += Z[i_row, j_row] * Zk[j]
            Y_buf_local[j] += Z[i_row, j_row] * Zk[i]
        end
    end
    Z[k, k] += dot(Y_buf_local, Zk)
    copyto!(Zk, -Y_buf_local)
end

function selinv_simplicial(F::SparseArrays.CHOLMOD.Factor; depermute = false)
    s = unsafe_load(pointer(F))
    if Bool(s.is_ll)
        Z = sparse(F.L)
        LL_simplicial_to_LDL!(Z)
    else
        ZD = sparse(F.LD)
        Z, d = SparseArrays.CHOLMOD.getLd!(ZD)
        for k in axes(Z, 2)
            Z[k, k] = 1 / d[k]
        end
    end

    _selinv_simplicial_Z!(Z)

    p = F.p
    if depermute
        # Symmetric(Z)[invperm(p), invperm(p)], but much faster
        # ... at the cost of more memory
        rows, cols, vals = findnz(sparse(Symmetric(Z)))
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

function _selinv_simplicial_Z!(Z::SparseMatrixCSC)
    N = size(Z, 2)
    Y_buf = zeros(N - 1)
    for j = (N-1):-1:1
        update_col!(Y_buf, Z, j)
    end
end
