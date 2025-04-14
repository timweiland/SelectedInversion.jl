import Base: size, IndexStyle, getindex
using SparseArrays

export SupernodalMatrix
export val_range, col_range, get_rows, get_row_col_idcs, get_max_sup_size
export get_Sj, partition_Sj
export get_chunk, get_split_chunk
export LL_to_LDL!

struct SupernodalMatrix <: AbstractArray{Float64, 2}
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

    function SupernodalMatrix(N, M, super_to_col, super_to_vals, super_to_rows, vals, rows, max_super_rows)
        n_super = length(super_to_col) - 1

        col_to_super = Vector{Int}(undef, M)
        cur_start = 1
        for s in 1:n_super
            cur_stop = super_to_col[s+1]
            col_to_super[cur_start:cur_stop] .= s
            cur_start = cur_stop + 1
        end

        return new(N, M, n_super, super_to_col, col_to_super, super_to_vals, super_to_rows, vals, rows, max_super_rows)
    end
end

function SupernodalMatrix(F::SparseArrays.CHOLMOD.Factor)
    s = unsafe_load(pointer(F))
    if !Bool(s.is_super)
        throw(ArgumentError("Expected supernodal Cholesky decomposition."))
    end
    N, M = size(F)
    n_super = Int(s.nsuper)
    max_super_rows = Int(s.maxesize)
    conv = p -> Base.unsafe_convert(Ptr{Int}, p)
    conv_float = p -> Base.unsafe_convert(Ptr{Float64}, p)
    super_to_col = copy(unsafe_wrap(Array, conv(s.super), n_super+1))
    super_to_vals = copy(unsafe_wrap(Array, conv(s.px), n_super+1))
    super_to_rows = copy(unsafe_wrap(Array, conv(s.pi), n_super+1))
    rows = copy(unsafe_wrap(Array, conv(s.s), Int(s.ssize)))
    vals = copy(unsafe_wrap(Array, conv_float(s.x), Int(s.xsize)))

    return SupernodalMatrix(N, M, super_to_col, super_to_vals, super_to_rows, vals, rows, max_super_rows)
end

Base.size(S::SupernodalMatrix) = (S.N, S.M)
Base.IndexStyle(::Type{<:SupernodalMatrix}) = IndexCartesian()

function val_range(S::SupernodalMatrix, sup_idx::Int)
    return (S.super_to_vals[sup_idx]+1):(S.super_to_vals[sup_idx+1])
end

function col_range(S::SupernodalMatrix, sup_idx::Int)
    return (S.super_to_col[sup_idx]+1):S.super_to_col[sup_idx+1]
end

function get_max_sup_size(S::SupernodalMatrix)
    if S.n_super == 1
        return length(col_range(S, 1))
    end
    return maximum(diff(S.super_to_col)[1:(end-1)])
end

function get_chunk(S::SupernodalMatrix, sup_idx::Int)
    col_rng = col_range(S, sup_idx)
    vals_rng = val_range(S, sup_idx)
    chunk = @view(S.vals[vals_rng])
    N_cols = length(col_rng)
    rows_rng = (S.super_to_rows[sup_idx]+1):S.super_to_rows[sup_idx+1]
    N_rows = length(rows_rng)
    return reshape(chunk, (N_rows, N_cols))
end

function get_rows(S::SupernodalMatrix, sup_idx::Int)
    rows_rng = (S.super_to_rows[sup_idx]+1):S.super_to_rows[sup_idx+1]
    return @view(S.rows[rows_rng])
end

function get_row_col_idcs(S::SupernodalMatrix, sup_idx::Int)
    return (get_rows(S, sup_idx) .+ 1, col_range(S, sup_idx))
end

function get_Sj(S::SupernodalMatrix, sup_idx::Int)
    col_rng = col_range(S, sup_idx)
    return get_rows(S, sup_idx)[(length(col_rng) + 1):end]
end

function partition_Sj(S::SupernodalMatrix, Sj)
    if length(Sj) == 0
        return Vector{Int64}[]
    end
    blocks = Vector{Vector{Int64}}(undef, 0)
    cur_block = nothing
    cur_sup = nothing
    last_row = 0

    for row in Sj
        sup = S.col_to_super[row + 1]

        if cur_sup === nothing
            cur_sup = sup
            cur_block = [row]
        elseif (sup === cur_sup) && (row == last_row + 1)
            push!(cur_block, row)
        else
            push!(blocks, cur_block)
            cur_sup = sup
            cur_block = [row]
        end
        last_row = row
    end
    push!(blocks, cur_block)
    return blocks
end

function Base.getindex(S::SupernodalMatrix, I::Vararg{Int, 2})
    i, j = I
    sup_idx = S.col_to_super[j]
    rows_rng = get_rows(S, sup_idx)
    row_idx = searchsorted(rows_rng, i - 1)
    if length(row_idx) != 1
        return 0.0
    end
    row_idx = row_idx[1]
    col_idx = j - S.super_to_col[sup_idx]
    return get_chunk(S, sup_idx)[row_idx, col_idx]
end

function get_diagonal_block(chunk)
    N_col = size(chunk, 2)
    return @view(chunk[1:N_col, 1:end])
end

function get_split_chunk(S::SupernodalMatrix, sup_idx::Int)
    chunk = get_chunk(S, sup_idx)
    diag_block = get_diagonal_block(chunk)
    if sup_idx == S.n_super
        return diag_block, nothing
    end
    off_diag_start = size(chunk, 2) + 1
    off_diag_block = @view(chunk[off_diag_start:end, 1:end])
    return diag_block, off_diag_block
end

function LL_to_LDL!(S::SupernodalMatrix)
    for j in 1:S.n_super
        diag, off_diag = get_split_chunk(S, j)
        LinearAlgebra.LAPACK.trtri!('L', 'N', diag)
        if j < S.n_super
            off_diag .= off_diag * diag
        end
        diag .= diag' * diag
    end
end
