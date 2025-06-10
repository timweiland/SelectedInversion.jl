import Base: size, IndexStyle, getindex
using SparseArrays
import SparseArrays: sparse
import LinearAlgebra: diag

export SupernodalMatrix
export val_range, col_range, get_rows, get_row_col_idcs, get_max_sup_size
export get_Sj, partition_Sj
export get_chunk, get_split_chunk

"""
    SupernodalMatrix

Represents a sparse block lower triangular matrix with a supernodal layout.
A supernode is a set of contiguous columns with identical sparsity pattern below
the triangular block at the top.
For each supernode, the corresponding nonzero entries are stored in a dense
chunk.
This enables us to use BLAS for operations on these chunks, so we combine the
strengths of sparse and dense matrices.

# Fields
- `N::Int`: Number of rows
- `M::Int`: Number of columns
- `n_super::Int`: Number of supernodes
- `super_to_col::Vector{Int}`: Start/end column of each supernode.
                               Length `n_super + 1`.
- `col_to_super::Vector{Int}`: Maps each column to its supernode index.
                               Length `M`.
- `super_to_vals::Vector{Int}`: Start/end indices of each supernode into `vals`.
                                Length `n_super + 1`.
- `super_to_rows::Vector{Int}`: Start/end indices of each supernode into `rows`.
                                Length `n_super + 1`.
- `vals::Vector{Float64}`: Nonzero values.
- `rows::Vector{Int}`: Row indices. CAREFUL: These are zero-indexed!
- `max_super_rows::Int`: Maximum number of rows below the triangular block in a
                         supernode chunk.
- `transposed_chunks::Bool`: Whether to store the transpose of chunks, such that
                             the first axis in the chunk corresponds to the
                             columns in the supernode.
- `symmetric_access::Bool`: Whether to enforce symmetry when accessing entries.
- `invperm::Vector{Int}`: Permutation to apply before accessing entries
                          when `depermuted_access == true`.
- `depermuted_access::Bool`: Whether to apply an inverse permutation before
                             accessing entries.
"""
struct SupernodalMatrix <: AbstractArray{Float64,2}
    N::Int
    M::Int
    n_super::Int
    super_to_col::Vector{Int}
    col_to_super::Vector{Int}
    super_to_vals::Vector{Int}
    super_to_rows::Vector{Int}
    vals::Vector{Float64}
    rows::Vector{Int}
    max_super_rows::Int
    transposed_chunks::Bool
    symmetric_access::Bool
    invperm::Vector{Int}
    depermuted_access::Bool

    function SupernodalMatrix(
        N,
        M,
        super_to_col,
        super_to_vals,
        super_to_rows,
        vals,
        rows,
        max_super_rows,
        invperm;
        transpose_chunks = false,
        symmetric_access = false,
        depermuted_access = false,
    )
        n_super = length(super_to_col) - 1

        col_to_super = Vector{Int}(undef, M)
        cur_start = 1
        for s = 1:n_super
            cur_stop = super_to_col[s+1]
            col_to_super[cur_start:cur_stop] .= s
            cur_start = cur_stop + 1
        end

        if transpose_chunks
            _transpose_chunks!(vals, super_to_vals, super_to_col)
        end

        return new(
            N,
            M,
            n_super,
            super_to_col,
            col_to_super,
            super_to_vals,
            super_to_rows,
            vals,
            rows,
            max_super_rows,
            transpose_chunks,
            symmetric_access,
            invperm,
            depermuted_access,
        )
    end
end

"""
    SupernodalMatrix(
        F::SparseArrays.CHOLMOD.Factor;
        transpose_chunks = false,
        symmetric_access = false,
        depermuted_access = false,
    )

Construct a `SupernodalMatrix` from a supernodal Cholesky factorization.

Keyword arguments are explained in the `SupernodalMatrix` docstring.
"""
function SupernodalMatrix(
    F::SparseArrays.CHOLMOD.Factor;
    transpose_chunks = false,
    symmetric_access = false,
    depermuted_access = false,
)
    s = unsafe_load(pointer(F))
    if !Bool(s.is_super)
        throw(ArgumentError("Expected supernodal Cholesky decomposition."))
    end
    N, M = size(F)
    n_super = Int(s.nsuper)
    max_super_rows = Int(s.maxesize)
    conv = p -> Base.unsafe_convert(Ptr{Int}, p)
    conv_float = p -> Base.unsafe_convert(Ptr{Float64}, p)
    super_to_col = copy(unsafe_wrap(Array, conv(s.super), n_super + 1))
    super_to_vals = copy(unsafe_wrap(Array, conv(s.px), n_super + 1))
    super_to_rows = copy(unsafe_wrap(Array, conv(s.pi), n_super + 1))
    rows = copy(unsafe_wrap(Array, conv(s.s), Int(s.ssize)))
    vals = copy(unsafe_wrap(Array, conv_float(s.x), Int(s.xsize)))

    return SupernodalMatrix(
        N,
        M,
        super_to_col,
        super_to_vals,
        super_to_rows,
        vals,
        rows,
        max_super_rows,
        invperm(F.p),
        transpose_chunks = transpose_chunks,
        symmetric_access = symmetric_access,
        depermuted_access = depermuted_access,
    )
end

Base.size(S::SupernodalMatrix) = (S.N, S.M)
Base.IndexStyle(::Type{<:SupernodalMatrix}) = IndexCartesian()

"""
    val_range(S::SupernodalMatrix, sup_idx::Int)

Get the range of indices of supernode `sup_idx` into `S.vals`.
"""
function val_range(S::SupernodalMatrix, sup_idx::Int)
    return (S.super_to_vals[sup_idx]+1):(S.super_to_vals[sup_idx+1])
end

"""
    col_range(S::SupernodalMatrix, sup_idx::Int)

Get the range of columns of supernode `sup_idx`.
"""
function col_range(S::SupernodalMatrix, sup_idx::Int)
    return (S.super_to_col[sup_idx]+1):S.super_to_col[sup_idx+1]
end

"""
    get_max_sup_size(S::SupernodalMatrix)

Get the maximum number of columns of any supernode.
"""
function get_max_sup_size(S::SupernodalMatrix)
    if S.n_super == 1
        return length(col_range(S, 1))
    end
    return maximum(diff(S.super_to_col)[1:(end-1)])
end

"""
    get_chunk(S::SupernodalMatrix, sup_idx::Int)

Get the dense chunk corresponding to supernode `sup_idx`.
Includes the triangular block at the top.
"""
function get_chunk(S::SupernodalMatrix, sup_idx::Int)
    col_rng = col_range(S, sup_idx)
    vals_rng = val_range(S, sup_idx)
    chunk = @view(S.vals[vals_rng])
    N_cols = length(col_rng)
    rows_rng = (S.super_to_rows[sup_idx]+1):S.super_to_rows[sup_idx+1]
    N_rows = length(rows_rng)
    if S.transposed_chunks
        return reshape(chunk, (N_cols, N_rows))
    else
        return reshape(chunk, (N_rows, N_cols))
    end
end

function _transpose_chunks!(vals_arr, super_to_vals, super_to_col)
    for sup_idx = 1:(length(super_to_vals)-1)
        rng_start = super_to_vals[sup_idx] + 1
        rng_stop = super_to_vals[sup_idx+1]
        N_cols = super_to_col[sup_idx+1] - super_to_col[sup_idx]
        N_rows = (rng_stop - rng_start + 1) ÷ N_cols
        chunk = reshape(vals_arr[rng_start:rng_stop], (N_rows, N_cols))
        copyto!(@view(vals_arr[rng_start:rng_stop]), vec(chunk'))
    end
end

"""
    get_rows(S::SupernodalMatrix, sup_idx::Int)

Get the row indices corresponding to supernode `sup_idx`.
CAREFUL: These are zero-indexed!
"""
function get_rows(S::SupernodalMatrix, sup_idx::Int)
    rows_rng = (S.super_to_rows[sup_idx]+1):S.super_to_rows[sup_idx+1]
    return @view(S.rows[rows_rng])
end

"""
    get_row_col_idcs(S::SupernodalMatrix, sup_idx::Int)

Get the row and column indices corresponding to supernode `sup_idx`.
Both sets of indices are one-indexed.
"""
function get_row_col_idcs(S::SupernodalMatrix, sup_idx::Int)
    return (get_rows(S, sup_idx) .+ 1, col_range(S, sup_idx))
end

"""
    get_Sj(S::SupernodalMatrix, sup_idx::Int)

Get the row indices *below the triangular block* of supernode `sup_idx`.
CAREFUL: These are zero-indexed!
"""
function get_Sj(S::SupernodalMatrix, sup_idx::Int)
    col_rng = col_range(S, sup_idx)
    return get_rows(S, sup_idx)[(length(col_rng)+1):end]
end

"""
    partition_Sj(S::SupernodalMatrix, Sj)

Partition the output of `get_Sj` into contiguous subsets where each subset is
fully contained in one supernode.
"""
function partition_Sj(S::SupernodalMatrix, Sj)
    if length(Sj) == 0
        return UnitRange{Int64}[]
    end
    blocks = Vector{UnitRange{Int64}}(undef, 0)
    cur_block = nothing
    cur_sup = nothing
    last_row = 0

    for row in Sj
        sup = S.col_to_super[row+1]

        if cur_sup === nothing
            cur_sup = sup
            cur_block = [row]
        elseif (sup === cur_sup) && (row == last_row + 1)
            push!(cur_block, row)
        else
            push!(blocks, cur_block[1]:cur_block[end])
            cur_sup = sup
            cur_block = [row]
        end
        last_row = row
    end
    push!(blocks, cur_block[1]:cur_block[end])
    return blocks
end

function Base.getindex(S::SupernodalMatrix, I::Vararg{Int,2})
    i, j = I
    if S.depermuted_access
        i, j = S.invperm[[i, j]]
    end
    sup_idx = S.col_to_super[j]
    if i < col_range(S, sup_idx)[1] # Upper triangle access
        if S.symmetric_access
            sup_idx = S.col_to_super[i]
            col = i
            row = j - 1 # Zero-indexed
        else
            return 0.0
        end
    else
        col = j
        row = i - 1 # Zero-indexed
    end
    rows_rng = get_rows(S, sup_idx)
    row_idx = searchsorted(rows_rng, row)
    if length(row_idx) != 1
        return 0.0
    end
    row_idx = row_idx[1]
    col_idx = col - S.super_to_col[sup_idx]
    if S.transposed_chunks
        return get_chunk(S, sup_idx)[col_idx, row_idx]
    else
        return get_chunk(S, sup_idx)[row_idx, col_idx]
    end
end

function _get_diagonal_block(chunk; transpose = false)
    if transpose
        N_col = size(chunk, 1)
        return @view(chunk[:, 1:N_col])
    else
        N_col = size(chunk, 2)
        return @view(chunk[1:N_col, :])
    end
end

"""
    get_split_chunk(S::SupernodalMatrix, sup_idx::Int)

Get the chunk corresponding to supernode `sup_idx`, split into the diagonal /
lower triangular block at the top, and the remaining block below it.
"""
function get_split_chunk(S::SupernodalMatrix, sup_idx::Int)
    chunk = get_chunk(S, sup_idx)
    diag_block = _get_diagonal_block(chunk; transpose = S.transposed_chunks)
    if sup_idx == S.n_super
        return diag_block, nothing
    end

    if S.transposed_chunks
        off_diag_start = size(chunk, 1) + 1
        off_diag_block = @view(chunk[:, off_diag_start:end])
    else
        off_diag_start = size(chunk, 2) + 1
        off_diag_block = @view(chunk[off_diag_start:end, :])
    end
    return diag_block, off_diag_block
end

function LL_to_LDL!(S::SupernodalMatrix)
    for j = 1:S.n_super
        diag, off_diag = get_split_chunk(S, j)
        if S.transposed_chunks
            if j < S.n_super
                LinearAlgebra.LAPACK.trtrs!('U', 'N', 'N', diag, off_diag)
            end
            LinearAlgebra.LAPACK.potri!('U', diag)
            diag .= Symmetric(diag, :U)
        else
            LinearAlgebra.LAPACK.trtri!('L', 'N', diag)
            if j < S.n_super
                off_diag .= off_diag * diag
            end
            diag .= Symmetric(diag' * diag)
        end
    end
end

function sparse(S::SupernodalMatrix)
    Vs = S.vals
    Is = Int64[]
    Js = Int64[]

    for sup_idx = 1:S.n_super
        col_rng = col_range(S, sup_idx)
        rows = get_rows(S, sup_idx) .+ 1
        if S.transposed_chunks
            N_col = length(col_rng)
            for row in rows
                append!(Is, repeat([row], N_col))
                append!(Js, col_rng)
            end
        else
            N_rows = length(rows)
            for col in col_rng
                append!(Is, rows)
                append!(Js, repeat([col], N_rows))
            end
        end
    end
    M = sparse(Is, Js, Vs)
    if S.symmetric_access
        M = sparse(Symmetric(M, :L))
    end
    if S.depermuted_access
        M = permute(M, S.invperm, S.invperm)
    end
    return M
end

function diag(S::SupernodalMatrix)
    res = zeros(minimum(size(S)))
    cur_chunk_start = 1
    for sup_idx = 1:S.n_super
        cur_diag = diag(get_split_chunk(S, sup_idx)[1])
        cur_rng = range(start = cur_chunk_start, length = length(cur_diag))
        copyto!(@view(res[cur_rng]), cur_diag)
        cur_chunk_start += length(cur_diag)
    end
    if S.depermuted_access
        res = res[S.invperm]
    end
    return res
end
