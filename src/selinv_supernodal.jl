using SparseArrays
using LinearAlgebra

export selinv_supernodal

const BlasInt = LinearAlgebra.BLAS.BlasInt

# ── Direct BLAS/LAPACK ccalls with pointer+offset ────────────────────────────
# These avoid creating SubArray/ReshapedArray wrappers entirely.

# C = α*B*A + β*C where A is symmetric (side='R', uplo='L')
# A at pA with lda, B at pB with ldb, C at pC with ldc
@inline function _dsymm!(m, n, α, pA, lda, pB, ldb, β, pC, ldc)
    ccall((BLAS.@blasfunc(dsymm_), BLAS.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
         Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ref{BlasInt},
         Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
         Clong, Clong),
        'R', 'L', m, n, α, pA, lda, pB, ldb, β, pC, ldc, 1, 1)
    return nothing
end

# C = α*A*B' + α*B*A' + β*C where C is symmetric (uplo='U', trans='N')
@inline function _dsyr2k!(n, k, α, pA, lda, pB, ldb, β, pC, ldc)
    ccall((BLAS.@blasfunc(dsyr2k_), BLAS.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
         Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
         Ptr{Float64}, Ref{BlasInt},
         Ref{Float64}, Ptr{Float64}, Ref{BlasInt},
         Clong, Clong),
        'U', 'N', n, k, α, pA, lda, pB, ldb, β, pC, ldc, 1, 1)
    return nothing
end

# Solve U*X = B in-place (B overwritten with X)
@inline function _dtrtrs!(n, nrhs, pA, lda, pB, ldb)
    info = Ref{BlasInt}(0)
    ccall((BLAS.@blasfunc(dtrtrs_), BLAS.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
         Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt},
         Clong, Clong, Clong),
        'U', 'N', 'N', n, nrhs, pA, lda, pB, ldb, info, 1, 1, 1)
    return nothing
end

# Compute inverse from Cholesky factor (upper triangle)
@inline function _dpotri!(n, pA, lda)
    info = Ref{BlasInt}(0)
    ccall((BLAS.@blasfunc(dpotri_), BLAS.libblastrampoline), Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt},
         Clong),
        'U', n, pA, lda, info, 1)
    return nothing
end

# Symmetrize: copy upper triangle to lower triangle in a chunk stored
# in vals at offset with leading dimension lda
@inline function _symmetrize!(vals, offset, n, lda)
    @inbounds for j in 1:n
        for i in (j + 1):n
            # vals[offset + (j-1)*lda + i - 1] = vals[offset + (i-1)*lda + j - 1]
            vals[offset + (j - 1) * lda + i] = vals[offset + (i - 1) * lda + j]
        end
    end
    return nothing
end

# Copy -Y into Z_Sj_j_T region of vals
@inline function _negate_copy!(dst, dst_offset, dst_lda, src, src_lda, m, n)
    @inbounds for j in 1:n
        for i in 1:m
            dst[dst_offset + (j - 1) * dst_lda + i] = -src[(j - 1) * src_lda + i]
        end
    end
    return nothing
end

# ── Helpers ──────────────────────────────────────────────────────────────────

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
    vals = Z.vals

    @inbounds for j in 1:n_blocks
        block_j = Sj_blocks[j]
        first_col_j = block_j[1] + 1
        K = Z.col_to_super[first_col_j]

        K_vals_base = Z.super_to_vals[K]
        K_n_cols = Z.super_to_col[K + 1] - Z.super_to_col[K]
        K_col_start = first_col_j - Z.super_to_col[K]

        R₁ = indmap[first(block_j)]:indmap[last(block_j)]

        # Diagonal sub-block
        for (mj, rj) in enumerate(R₁)
            for (mi, ri) in enumerate(R₁)
                c = K_col_start + mj - 1
                r = K_col_start + mi - 1
                M[ri, rj] = vals[K_vals_base + (r - 1) * K_n_cols + c]
            end
        end

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

            R₂ = indmap[first(block_i)]:indmap[last(block_i)]

            for (mj, rj) in enumerate(R₁)
                c = K_col_start + mj - 1
                for (mi, ri) in enumerate(R₂)
                    r = Z_chunk_row_idx + mi - 1
                    M[ri, rj] = vals[K_vals_base + (r - 1) * K_n_cols + c]
                end
            end
        end
    end
    clear_indmap!(Sj_blocks, n_blocks, indmap)
    return
end

# ── Main algorithm ───────────────────────────────────────────────────────────

function selinv_supernodal(F::SparseArrays.CHOLMOD.Factor; depermute = false)
    Z = SupernodalMatrix(
        F;
        transpose_chunks = true,
        symmetric_access = true,
        depermuted_access = depermute,
    )

    max_sup_size = get_max_sup_size(Z)
    Y_T_buf = zeros(max_sup_size, Z.max_super_rows)
    M_buf = zeros(Z.max_super_rows, Z.max_super_rows)
    indmap = zeros(Int, size(Z, 1))
    blocks_buf = Vector{UnitRange{Int64}}(undef, Z.max_super_rows)
    lda_Y = size(Y_T_buf, 1)
    lda_M = size(M_buf, 1)

    GC.@preserve Z Y_T_buf M_buf begin
        pvals = pointer(Z.vals)
        pY = pointer(Y_T_buf)
        pM = pointer(M_buf)

        # LL_to_LDL: forward pass for cache locality
        for j in 1:Z.n_super
            offset = Z.super_to_vals[j]  # 0-indexed into vals
            n_cols = Z.super_to_col[j + 1] - Z.super_to_col[j]
            n_rows = Z.super_to_rows[j + 1] - Z.super_to_rows[j]
            n_Sj = n_rows - n_cols
            pchunk = pvals + offset * sizeof(Float64)

            if j < Z.n_super && n_Sj > 0
                _dtrtrs!(n_cols, n_Sj, pchunk, n_cols,
                    pchunk + n_cols * n_cols * sizeof(Float64), n_cols)
            end
            _dpotri!(n_cols, pchunk, n_cols)
            _symmetrize!(Z.vals, offset, n_cols, n_cols)
        end

        # Backward sweep
        for sup_idx in (Z.n_super - 1):-1:1
            Sj = get_Sj(Z, sup_idx)
            n_blocks = partition_Sj!(blocks_buf, Z, Sj)

            offset = Z.super_to_vals[sup_idx]  # 0-indexed
            sup_size = Z.super_to_col[sup_idx + 1] - Z.super_to_col[sup_idx]
            n_Sj = length(Sj)

            pZ_jj = pvals + offset * sizeof(Float64)
            pZ_off = pZ_jj + sup_size * sup_size * sizeof(Float64)

            # Gather M = Z[Sj, Sj]
            gather_update_matrix!(M_buf, Z, blocks_buf, n_blocks, indmap)

            # Y = Z_Sj_j_T * M  (symm! with M on the right)
            _dsymm!(sup_size, n_Sj, 1.0, pM, lda_M, pZ_off, sup_size, 0.0, pY, lda_Y)

            # Z_jj += 0.5*(Y*Z_Sj_j_T' + Z_Sj_j_T*Y')
            _dsyr2k!(sup_size, n_Sj, 0.5, pY, lda_Y, pZ_off, sup_size, 1.0, pZ_jj, sup_size)
            _symmetrize!(Z.vals, offset, sup_size, sup_size)

            # Z_Sj_j_T = -Y
            _negate_copy!(Z.vals, offset + sup_size * sup_size, sup_size,
                Y_T_buf, lda_Y, sup_size, n_Sj)
        end
    end

    return (Z = Z, p = F.p)
end
