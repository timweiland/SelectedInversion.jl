using SparseArrays
using LinearAlgebra

export selinv

function build_indmap(Sj_blocks, indmap)
    i = 1
    for block in Sj_blocks
        for row in block
            indmap[row] = i
            i += 1
        end
    end
end

function compute_Y(Y_buf, Z, off_diag_chunk, Sj_blocks, indmap)
    fill!(Y_buf, 0.)
    build_indmap(Sj_blocks, indmap)
    n_j = length(Sj_blocks)
    @inbounds for j in 1:n_j
        block_j = Sj_blocks[j]
        first_col_j = block_j[1] + 1
        K = Z.col_to_super[first_col_j]

        Z_chunk_j = get_chunk(Z, K)
        Z_jj_start = Z.super_to_col[K]
        Z_jj_rng = range(start=first_col_j - Z_jj_start, length=length(block_j))
        Z_jj = @view(Z_chunk_j[Z_jj_rng, Z_jj_rng])

        Z_rows = get_rows(Z, K)
        Z_chunk_row_idx = 1

        R₁ = view(indmap, block_j)

        L_Ij_j = @view(off_diag_chunk[R₁, 1:end])

        Y_R₁ = @view(Y_buf[R₁, 1:end])
        mul!(Y_R₁, Z_jj, L_Ij_j, 1, 1)
        @inbounds for i in (j+1):n_j
            block_i = Sj_blocks[i]

            while true
                row = Z_rows[Z_chunk_row_idx]
                if (indmap[row] != 0) && row == block_i[1]
                    break
                end
                Z_chunk_row_idx += 1
            end
            Z_chunk_rng = range(start=Z_chunk_row_idx, length=length(block_i))

            R₂ = view(indmap, block_i)

            Z_ij = @view(Z_chunk_j[Z_chunk_rng, Z_jj_rng])
            L_Ii_j = @view(off_diag_chunk[R₂, 1:end])

            Y_R₂ = @view(Y_buf[R₂, 1:end])
            mul!(Y_R₂, Z_ij, L_Ij_j, 1, 1)
            mul!(Y_R₁, Z_ij', L_Ii_j, 1, 1)
        end
    end
    fill!(indmap, 0)
end

function selinv(F::SparseArrays.CHOLMOD.Factor; depermute=false)
    #F_cp = copy(F)
    L = SupernodalMatrix(F)
    Z = SupernodalMatrix(F)
    LL_to_LDL!(Z)

    max_sup_size = get_max_sup_size(L)
    Y_buf = zeros(L.max_super_rows, max_sup_size)
    indmap = zeros(Int, size(L, 1))

    for sup_idx in (L.n_super-1):-1:1
        Sj = get_Sj(L, sup_idx)
        Sj_blocks = partition_Sj(L, Sj)

        Z_jj, Z_Sj_j = get_split_chunk(Z, sup_idx)
        sup_size = size(Z_jj, 2)

        Y_cur = @view(Y_buf[1:length(Sj), 1:sup_size])
        compute_Y(Y_cur, Z, Z_Sj_j, Sj_blocks, indmap)

        mul!(Z_jj, Y_cur', Z_Sj_j, 1, 1)
        Z_Sj_j .= -Y_cur
    end

    return Z
end
