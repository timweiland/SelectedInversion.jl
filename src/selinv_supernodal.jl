using SparseArrays
using LinearAlgebra

export selinv_supernodal

# Use gather+symm! when the separator has at least this many blocks.
# Below this, the O(n_blocks²) scattered mul! calls are cheap enough
# that the gather copy overhead isn't worth it.
const GATHER_MIN_BLOCKS = 8

function build_indmap(Sj_blocks, n_blocks, indmap)
    i = 1
    @inbounds for b in 1:n_blocks
        block = Sj_blocks[b]
        for row in block
            indmap[row] = i
            i += 1
        end
    end
    return
end

function clear_indmap!(Sj_blocks, n_blocks, indmap)
    @inbounds for b in 1:n_blocks
        for row in Sj_blocks[b]
            indmap[row] = 0
        end
    end
    return
end

"""
    partition_Sj!(blocks_buf, S, Sj) -> Int

Non-allocating version of `partition_Sj`. Writes blocks into `blocks_buf`
and returns the number of blocks.
"""
function partition_Sj!(blocks_buf, S::SupernodalMatrix, Sj)
    length(Sj) == 0 && return 0

    n_blocks = 0
    block_start = Sj[1]
    block_end = Sj[1]
    cur_sup = S.col_to_super[Sj[1] + 1]

    @inbounds for idx in 2:length(Sj)
        row = Sj[idx]
        sup = S.col_to_super[row + 1]

        if sup == cur_sup && row == block_end + 1
            block_end = row
        else
            n_blocks += 1
            blocks_buf[n_blocks] = block_start:block_end
            cur_sup = sup
            block_start = row
            block_end = row
        end
    end
    n_blocks += 1
    blocks_buf[n_blocks] = block_start:block_end
    return n_blocks
end

function gather_update_matrix!(M, Z, Sj_blocks, n_blocks, indmap)
    build_indmap(Sj_blocks, n_blocks, indmap)

    @inbounds for j in 1:n_blocks
        block_j = Sj_blocks[j]
        first_col_j = block_j[1] + 1
        K = Z.col_to_super[first_col_j]

        Z_chunk = get_chunk(Z, K)
        K_start = Z.super_to_col[K]
        col_rng = range(start = first_col_j - K_start, length = length(block_j))

        R₁ = indmap[block_j][1]:indmap[block_j][end]

        # Diagonal sub-block
        M[R₁, R₁] .= @view(Z_chunk[col_rng, col_rng])

        # Off-diagonal sub-blocks (lower triangle of M)
        Z_rows = get_rows(Z, K)
        Z_chunk_row_idx = 1

        @inbounds for i in (j + 1):n_blocks
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

            @view(M[R₂, R₁]) .= @view(Z_chunk[col_rng, row_rng])'
        end
    end
    clear_indmap!(Sj_blocks, n_blocks, indmap)
    return
end

function compute_Y_scattered!(Y_buf, Z, off_diag_chunk, Sj_blocks, n_blocks, indmap)
    fill!(Y_buf, 0.0)
    build_indmap(Sj_blocks, n_blocks, indmap)
    @inbounds for j in 1:n_blocks
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
        @inbounds for i in (j + 1):n_blocks
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
    clear_indmap!(Sj_blocks, n_blocks, indmap)
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
    blocks_buf = Vector{UnitRange{Int64}}(undef, Z.max_super_rows)

    GC.@preserve Z begin
        for sup_idx in (Z.n_super - 1):-1:1
            Sj = get_Sj(Z, sup_idx)
            n_blocks = partition_Sj!(blocks_buf, Z, Sj)

            chunk = get_chunk(Z, sup_idx)
            sup_size = size(chunk, 1)  # n_cols for transposed layout
            Z_jj = @view(chunk[:, 1:sup_size])
            Z_Sj_j_T = @view(chunk[:, (sup_size + 1):size(chunk, 2)])
            n_Sj = length(Sj)

            Y = @view(Y_T_buf[1:sup_size, 1:n_Sj])

            if n_blocks >= GATHER_MIN_BLOCKS
                M = @view(M_buf[1:n_Sj, 1:n_Sj])
                gather_update_matrix!(M, Z, blocks_buf, n_blocks, indmap)
                BLAS.symm!('R', 'L', 1.0, M, Z_Sj_j_T, 0.0, Y)
            else
                compute_Y_scattered!(Y, Z, Z_Sj_j_T, blocks_buf, n_blocks, indmap)
            end

            # Z_jj += Y * off_diag' (symmetric rank-2k update, upper triangle only)
            BLAS.syr2k!('U', 'N', 0.5, Y, Z_Sj_j_T, 1.0, Z_jj)
            Z_jj .= Symmetric(Z_jj, :U)

            Z_Sj_j_T .= -Y
        end
    end

    return (Z = Z, p = F.p)
end
