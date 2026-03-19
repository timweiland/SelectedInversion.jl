using SparseArrays
using LinearAlgebra

export selinv_supernodal

# Use gather+symm! when the separator has at least this many blocks.
# Below this, the O(n_blocks²) scattered mul! calls are cheap enough
# that the gather copy overhead isn't worth it.
const GATHER_MIN_BLOCKS = 8

function build_indmap(Sj_blocks, indmap)
    i = 1
    for block in Sj_blocks
        for row in block
            indmap[row] = i
            i += 1
        end
    end
    return
end

"""
    gather_update_matrix!(M, Z, Sj_blocks, indmap)

Assemble the update matrix M = Z[Sj, Sj] into a contiguous pre-allocated buffer
by copying sub-blocks from the already-computed supernodal chunks.

Only the lower triangle of M is filled (sufficient for `BLAS.symm!`).
"""
function gather_update_matrix!(M, Z, Sj_blocks, indmap)
    build_indmap(Sj_blocks, indmap)
    n_j = length(Sj_blocks)

    @inbounds for j in 1:n_j
        block_j = Sj_blocks[j]
        first_col_j = block_j[1] + 1
        K = Z.col_to_super[first_col_j]

        Z_chunk = get_chunk(Z, K)
        K_start = Z.super_to_col[K]
        col_rng = range(start = first_col_j - K_start, length = length(block_j))

        R₁ = indmap[block_j][1]:indmap[block_j][end]

        # Diagonal sub-block: M[R₁, R₁] = Z[block_j, block_j]
        M[R₁, R₁] .= @view(Z_chunk[col_rng, col_rng])

        # Off-diagonal sub-blocks (lower triangle of M)
        Z_rows = get_rows(Z, K)
        Z_chunk_row_idx = 1

        @inbounds for i in (j + 1):n_j
            block_i = Sj_blocks[i]

            while true
                row = Z_rows[Z_chunk_row_idx]
                if (indmap[row] != 0) && row == block_i[1]
                    break
                end
                Z_chunk_row_idx += 1
            end
            row_rng = range(start = Z_chunk_row_idx, length = length(block_i))

            R₂ = indmap[block_i][1]:indmap[block_i][end]

            # chunk[col_rng, row_rng] stores Z[block_i_rows, block_j_cols] (transposed layout)
            # M[R₂, R₁] = Z[block_i, block_j] = transpose of chunk[col_rng, row_rng]
            @view(M[R₂, R₁]) .= @view(Z_chunk[col_rng, row_rng])'
        end
    end
    fill!(indmap, 0)
    return
end

"""
    compute_Y_scattered!(Y_buf, Z, off_diag_chunk, Sj_blocks, indmap)

Compute Y = off_diag * Z[Sj, Sj] using scattered block-by-block mul! calls.
Faster for tiny supernodes where gather overhead dominates.
"""
function compute_Y_scattered!(Y_buf, Z, off_diag_chunk, Sj_blocks, indmap)
    fill!(Y_buf, 0.0)
    build_indmap(Sj_blocks, indmap)
    n_j = length(Sj_blocks)
    @inbounds for j in 1:n_j
        block_j = Sj_blocks[j]
        first_col_j = block_j[1] + 1
        K = Z.col_to_super[first_col_j]

        Z_chunk_j = get_chunk(Z, K)
        Z_jj_start = Z.super_to_col[K]
        Z_jj_rng = range(start = first_col_j - Z_jj_start, length = length(block_j))
        Z_jj = @view(Z_chunk_j[Z_jj_rng, Z_jj_rng])

        Z_rows = get_rows(Z, K)
        Z_chunk_row_idx = 1

        R₁ = indmap[block_j][1]:indmap[block_j][end]

        L_Ij_j_T = @view(off_diag_chunk[:, R₁])

        Y_R₁ = @view(Y_buf[:, R₁])
        mul!(Y_R₁, L_Ij_j_T, Z_jj, 1, 1)
        @inbounds for i in (j + 1):n_j
            block_i = Sj_blocks[i]

            while true
                row = Z_rows[Z_chunk_row_idx]
                if (indmap[row] != 0) && row == block_i[1]
                    break
                end
                Z_chunk_row_idx += 1
            end
            Z_chunk_rng = range(start = Z_chunk_row_idx, length = length(block_i))

            R₂ = indmap[block_i][1]:indmap[block_i][end]

            Z_ij_T = @view(Z_chunk_j[Z_jj_rng, Z_chunk_rng])
            L_Ii_j_T = @view(off_diag_chunk[:, R₂])

            Y_R₂ = @view(Y_buf[:, R₂])

            mul!(Y_R₂, L_Ij_j_T, Z_ij_T, 1, 1)
            mul!(Y_R₁, L_Ii_j_T, Z_ij_T', 1, 1)
        end
    end
    fill!(indmap, 0)
    return
end

function selinv_supernodal(F::SparseArrays.CHOLMOD.Factor; depermute = false)
    Z = SupernodalMatrix(
        F;
        transpose_chunks = true,
        symmetric_access = true,
        depermuted_access = depermute,
    )
    LL_to_LDL!(Z)

    max_sup_size = get_max_sup_size(Z)
    Y_T_buf = zeros(max_sup_size, Z.max_super_rows)
    M_buf = zeros(Z.max_super_rows, Z.max_super_rows)
    indmap = zeros(Int, size(Z, 1))

    for sup_idx in (Z.n_super - 1):-1:1
        Sj = get_Sj(Z, sup_idx)
        Sj_blocks = partition_Sj(Z, Sj)

        Z_jj, Z_Sj_j_T = get_split_chunk(Z, sup_idx)
        sup_size = size(Z_jj, 1)
        n_Sj = length(Sj)

        Y = @view(Y_T_buf[1:sup_size, 1:n_Sj])

        n_blocks = length(Sj_blocks)
        if n_blocks >= GATHER_MIN_BLOCKS
            # Gather + BLAS path: one symm! call instead of O(n_blocks²) mul! calls
            M = @view(M_buf[1:n_Sj, 1:n_Sj])
            gather_update_matrix!(M, Z, Sj_blocks, indmap)
            BLAS.symm!('R', 'L', 1.0, M, Z_Sj_j_T, 0.0, Y)
        else
            # Scattered path: fewer blocks means mul! overhead is low
            compute_Y_scattered!(Y, Z, Z_Sj_j_T, Sj_blocks, indmap)
        end

        # Z_jj += Y * off_diag' (symmetric rank-2k update, upper triangle only)
        BLAS.syr2k!('U', 'N', 0.5, Y, Z_Sj_j_T, 1.0, Z_jj)
        Z_jj .= Symmetric(Z_jj, :U)

        Z_Sj_j_T .= -Y
    end

    return (Z = Z, p = F.p)
end
