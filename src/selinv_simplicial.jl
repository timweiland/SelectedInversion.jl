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
            Z[k, k] = 1 / d^2
        end
    end

    N = size(Z, 2)
    Y_buf = zeros(N - 1)
    for j = (N-1):-1:1
        update_col!(Y_buf, Z, j)
    end

    p = F.p
    if depermute
        return (Z = permute(Z, invperm(p), invperm(p)), p = p)
    else
        return (Z = Z, p = p)
    end
end
