using SelectedInversion
using SparseArrays
using Test

export check_selinv, check_sparsity_pattern

function check_selinv(A_inv_approx::SupernodalMatrix, A_inv::AbstractMatrix)
    if size(A_inv_approx) != size(A_inv)
        return false
    end
    # Chunk access is permuted, so we need to permute the ground-truth as well
    p = invperm(A_inv_approx.invperm)
    A_inv = A_inv[p, p]
    for sup_idx = 1:A_inv_approx.n_super
        chunk = get_chunk(A_inv_approx, sup_idx)
        chunk_row_idcs, chunk_col_idcs = get_row_col_idcs(A_inv_approx, sup_idx)

        gt = A_inv[chunk_row_idcs, chunk_col_idcs]
        if A_inv_approx.transposed_chunks
            gt = gt'
        end

        if !(gt ≈ chunk)
            println("Violation at $(sup_idx)")
            return false
        end
    end
    return true
end

function check_selinv(A_inv_approx::SparseMatrixCSC, A_inv::AbstractMatrix)
    if size(A_inv_approx) != size(A_inv)
        return false
    end
    for j = 1:size(A_inv_approx, 2)
        nzrng = nzrange(A_inv_approx, j)
        vals = A_inv_approx.nzval[nzrng]
        rows = A_inv_approx.rowval[nzrng]
        if !(vals ≈ A_inv[rows, j])
            return false
        end
    end
    return true
end

function check_sparsity_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC)
    I_A, J_A, _ = findnz(A)
    I_B, J_B, _ = findnz(B)
    set_A = Set(zip(I_A, J_A))
    set_B = Set(zip(I_B, J_B))
    @test issubset(set_B, set_A)
end
