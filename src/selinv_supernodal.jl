using SparseArrays
using LinearAlgebra

export selinv_supernodal, ParallelConfig

"""
    ParallelConfig

Configuration for parallel selected inversion.

# Fields
- `enabled::Bool`: Whether to use parallel execution
- `min_level_size::Int`: Minimum supernodes at a level to trigger parallelization

# Performance Notes
Parallelization is most beneficial when:
- The matrix has many small supernodes (sparse factorization)
- `min_level_size` is tuned to avoid overhead for small batches (default: 8)

For matrices with few large supernodes, BLAS operations may already utilize
multiple threads, making additional parallelization counterproductive.
Consider setting `BLAS.set_num_threads(1)` if using parallel selected inversion.
"""
struct ParallelConfig
    enabled::Bool
    min_level_size::Int
end

"""
    ParallelConfig(enabled::Bool)

Create a ParallelConfig with default `min_level_size=8`.
A higher threshold reduces threading overhead for small parallel batches.
"""
ParallelConfig(enabled::Bool) = ParallelConfig(enabled, 8)

# Convenience constants
const PARALLEL_DEFAULT = ParallelConfig(true, 8)

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

function compute_Y(Y_buf, Z, off_diag_chunk, Sj_blocks, indmap)
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
    return fill!(indmap, 0)
end

"""
    _process_supernode!(Z, sup_idx, Y_T_buf, indmap)

Process a single supernode during the backward selected inversion sweep.
Updates Z in-place with the selected inverse entries for this supernode.
"""
function _process_supernode!(Z::SupernodalMatrix, sup_idx::Int, Y_T_buf::Matrix{Float64}, indmap::Vector{Int})
    Sj = get_Sj(Z, sup_idx)
    Sj_blocks = partition_Sj(Z, Sj)

    Z_jj, Z_Sj_j_T = get_split_chunk(Z, sup_idx)
    sup_size = size(Z_jj, 1)

    Y_T_cur = @view(Y_T_buf[1:sup_size, 1:length(Sj)])
    compute_Y(Y_T_cur, Z, Z_Sj_j_T, Sj_blocks, indmap)

    mul!(Z_jj, Y_T_cur, Z_Sj_j_T', 1, 1)
    Z_jj .= 0.5 * (Z_jj + Z_jj')  # Restore symmetry

    Z_Sj_j_T .= -Y_T_cur
    return nothing
end

"""
    _selinv_supernodal_sequential!(Z::SupernodalMatrix)

Perform sequential supernodal selected inversion backward sweep.
"""
function _selinv_supernodal_sequential!(Z::SupernodalMatrix)
    max_sup_size = get_max_sup_size(Z)
    Y_T_buf = zeros(max_sup_size, Z.max_super_rows)
    indmap = zeros(Int, size(Z, 1))

    for sup_idx in (Z.n_super - 1):-1:1
        _process_supernode!(Z, sup_idx, Y_T_buf, indmap)
    end
    return nothing
end

"""
    _selinv_supernodal_parallel!(Z::SupernodalMatrix, etree::SupernodalETree, config::ParallelConfig)

Perform parallel supernodal selected inversion using level-based scheduling.
Supernodes at the same level in the elimination tree are processed in parallel.
"""
function _selinv_supernodal_parallel!(Z::SupernodalMatrix, etree::SupernodalETree, config::ParallelConfig)
    max_sup_size = get_max_sup_size(Z)

    # Allocate buffers for maximum possible thread count (handles interactive threads)
    max_threads = Threads.maxthreadid()
    Y_T_bufs = [zeros(max_sup_size, Z.max_super_rows) for _ in 1:max_threads]
    indmaps = [zeros(Int, size(Z, 1)) for _ in 1:max_threads]

    # Process levels from root (level 0) toward leaves
    for level_idx in 1:etree.n_levels
        supernodes_at_level = etree.levels[level_idx]

        # Filter out root supernode (n_super) which doesn't need processing
        supernodes_to_process = filter(s -> s != Z.n_super, supernodes_at_level)

        if isempty(supernodes_to_process)
            continue
        end

        n_at_level = length(supernodes_to_process)

        if n_at_level >= config.min_level_size
            # Parallel execution for levels with enough work
            Threads.@threads for sup_idx in supernodes_to_process
                tid = Threads.threadid()
                _process_supernode!(Z, sup_idx, Y_T_bufs[tid], indmaps[tid])
            end
        else
            # Sequential for small levels (avoid threading overhead)
            for sup_idx in supernodes_to_process
                _process_supernode!(Z, sup_idx, Y_T_bufs[1], indmaps[1])
            end
        end
    end
    return nothing
end

"""
    selinv_supernodal(F::CHOLMOD.Factor; depermute=false, parallel=false)

Compute selected inverse using supernodal algorithm.

# Keyword arguments
- `depermute::Bool=false`: Apply inverse permutation to result
- `parallel::Union{Bool, ParallelConfig}=false`: Enable parallel execution
  - `false`: Sequential execution
  - `true`: Parallel with default configuration
  - `ParallelConfig(...)`: Custom parallel configuration
"""
function selinv_supernodal(
    F::SparseArrays.CHOLMOD.Factor;
    depermute::Bool = false,
    parallel::Union{Bool, ParallelConfig} = false
)
    # Convert Bool to ParallelConfig
    config = parallel isa Bool ? ParallelConfig(parallel) : parallel

    Z = SupernodalMatrix(
        F;
        transpose_chunks = true,
        symmetric_access = true,
        depermuted_access = depermute,
    )
    LL_to_LDL!(Z)

    # Dispatch based on configuration and thread availability
    if config.enabled && Threads.nthreads() > 1
        etree = build_supernodal_etree(Z)
        _selinv_supernodal_parallel!(Z, etree, config)
    else
        _selinv_supernodal_sequential!(Z)
    end

    return (Z = Z, p = F.p)
end
