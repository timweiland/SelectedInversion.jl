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

        R₁ = indmap[block_j][1]:indmap[block_j][end]

        L_Ij_j_T = @view(off_diag_chunk[:, R₁])

        Y_R₁ = @view(Y_buf[:, R₁])
        mul!(Y_R₁, L_Ij_j_T, Z_jj, 1, 1)
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

            R₂ = indmap[block_i][1]:indmap[block_i][end]

            Z_ij_T = @view(Z_chunk_j[Z_jj_rng, Z_chunk_rng])
            L_Ii_j_T = @view(off_diag_chunk[:, R₂])

            Y_R₂ = @view(Y_buf[:, R₂])

            mul!(Y_R₂, L_Ij_j_T, Z_ij_T, 1, 1)
            mul!(Y_R₁, L_Ii_j_T, Z_ij_T', 1, 1)
        end
    end
    fill!(indmap, 0)
end

function selinv(F::SparseArrays.CHOLMOD.Factor; depermute=false)
    Z = SupernodalMatrix(F; transpose_chunks=true)
    LL_to_LDL!(Z)

    max_sup_size = get_max_sup_size(Z)
    Y_T_buf = zeros(max_sup_size, Z.max_super_rows)
    indmap = zeros(Int, size(Z, 1))

    for sup_idx in (Z.n_super-1):-1:1
        Sj = get_Sj(Z, sup_idx)
        Sj_blocks = partition_Sj(Z, Sj)

        Z_jj, Z_Sj_j_T = get_split_chunk(Z, sup_idx)
        sup_size = size(Z_jj, 1)

        Y_T_cur = @view(Y_T_buf[1:sup_size, 1:length(Sj)])
        compute_Y(Y_T_cur, Z, Z_Sj_j_T, Sj_blocks, indmap)

        mul!(Z_jj, Y_T_cur, Z_Sj_j_T', 1, 1)
        Z_Sj_j_T .= -Y_T_cur
    end

    return Z
end
