# Selected inversion from a pivoted sparse LU factorization (UMFPACK, or KLU
# through the `SelectedInversionKLUExt` extension).
#
# The factors are read as `(R A)[P, Q] = L U` (see `lu_factors.jl`), with a row
# scaling `R`, independent row and column permutations `P` and `Q`, `L` unit
# lower and `U` upper triangular; KLU's block triangular form is folded into `U`
# when it is read. With `M = (R A)[P, Q]` and `X = M⁻¹`,
#
#     A⁻¹[Q[j], P[i]] = X[j, i] * r[P[i]],
#
# where `r` holds the row multipliers. `X` follows from `X L = U⁻¹` and
# `U X = L⁻¹` (scalar form of Algorithm 1 in Jacquelin, Lin & Yang, 2018,
# https://arxiv.org/abs/1708.04539), processing k = n, …, 1:
#
#     X[r, k] = -Σ_{i ∈ C(k)} X[r, i] L[i, k]             (r > k)
#     X[k, c] = -Σ_{j ∈ R(k)} U[k, j] X[j, c] / U[k, k]   (c > k)
#     X[k, k] = (1 - Σ_{j ∈ R(k)} U[k, j] X[j, k]) / U[k, k]
#
# with C(k) the stored rows of column k of L and R(k) the stored columns of row k
# of U (both strictly off the diagonal).
#
# Which entries of X are computed is decided symbolically. Starting from the
# requested entries and all diagonal entries, the set N is closed under the
# dependencies of the recurrences:
#
#   (a) (r, k) ∈ N, i ∈ C(k)      ⇒ (r, i) ∈ N
#   (b) (k, c) ∈ N, j ∈ R(k)      ⇒ (j, c) ∈ N
#   (d) (k, k) ∈ N, j ∈ R(k)      ⇒ (j, k) ∈ N
#
# Without numerical cancellation, the closure of the transposed pattern of `M`
# is the transposed pattern of `L + U`, the usual selected inverse. UMFPACK drops
# entries of `L` and `U` that cancel to exactly zero, so the closure is computed
# from the stored factors instead of assumed; it also covers requested entries
# outside the transposed pattern of `L + U` (for example the pattern of `A`
# itself when `A` is unsymmetric). The closure costs as many set operations as
# the numeric recurrence has multiply-adds.

export LUSelInvWorkspace, selinv!

"""
    LUSelInvWorkspace(F, B::SparseMatrixCSC)

Workspace for computing the entries of `A⁻¹` at the stored positions of `B` from
a sparse LU factorization `F` of a square matrix `A`: `F = lu(A)` (UMFPACK), or
`F = klu(A)` (KLU, after `using KLU`).

The constructor prepares the symbolic structure for the pivot sequence and factor
pattern of `F` and the pattern of `B`. Pass the workspace to
[`selinv!(dest, ws, F)`](@ref selinv!)
to compute the values, including after refactorizing `F` (`lu!` or `klu!`).

Every stored entry of `B` is a requested position, including explicitly stored
zeros; the values of `B` are ignored. `B` uses the original row and column
numbering of `A`.

A workspace is not safe for concurrent use. Give each task or thread its own
workspace.
"""
mutable struct LUSelInvWorkspace{Tv, Ti}
    n::Int
    factors::_LUFactors{Tv, Ti}
    # Factor structure the workspace was prepared for (UMFPACK layout, 0-based).
    prepared_Lp::Vector{Ti}
    prepared_Lj::Vector{Ti}
    prepared_Up::Vector{Ti}
    prepared_Ui::Vector{Ti}
    prepared_P::Vector{Ti}
    prepared_Q::Vector{Ti}
    # Strictly lower part of L by columns; L_src[t] indexes the native row storage.
    L_colptr::Vector{Int}
    L_rowval::Vector{Int}
    L_src::Vector{Int}
    L_val::Vector{Tv}
    # Strictly upper part of U by rows; U_src[t] indexes the native column storage.
    U_rowptr::Vector{Int}
    U_colval::Vector{Int}
    U_src::Vector{Int}
    U_val::Vector{Tv}
    U_diag_src::Vector{Int}
    U_diag::Vector{Tv}
    # Computed entries of X: strictly lower part by columns, strictly upper part
    # by rows. Values are stored in `X_val` as [lower; upper; diagonal].
    lo_colptr::Vector{Int}
    lo_rowval::Vector{Int}
    up_rowptr::Vector{Int}
    up_colval::Vector{Int}
    X_val::Vector{Tv}
    # Requested pattern (original numbering) and, per stored entry, its source
    # in X_val.
    B_colptr::Vector{Int}
    B_rowval::Vector{Int}
    B_src::Vector{Int}
    row_multipliers::Vector{Float64}
    preparations::Int
end

Base.size(ws::LUSelInvWorkspace) = (ws.n, ws.n)

LUSelInvWorkspace(F::_UmfpackLU{Tv, Ti}, B::SparseMatrixCSC) where {Tv, Ti} =
    _lu_workspace(F, B, Tv, Ti)

function _lu_workspace(F::Factorization, B::SparseMatrixCSC, ::Type{Tv}, ::Type{Ti}) where {Tv, Ti}
    n = size(F, 1)
    size(B) == size(F) || throw(
        DimensionMismatch("requested pattern has size $(size(B)), factorization has size $(size(F))"),
    )
    ws = LUSelInvWorkspace{Tv, Ti}(
        n, _LUFactors{Tv, Ti}(),
        Ti[], Ti[], Ti[], Ti[], Ti[], Ti[],
        Int[], Int[], Int[], Tv[],
        Int[], Int[], Int[], Tv[],
        Int[], Tv[],
        Int[], Int[], Int[], Int[], Tv[],
        Vector{Int}(SparseArrays.getcolptr(B)), Vector{Int}(rowvals(B)), Int[],
        Float64[], 0,
    )
    _read_factors!(ws.factors, F)
    _prepare!(ws)
    return ws
end

"""
    selinv(F::SparseArrays.UMFPACK.UmfpackLU, B::SparseMatrixCSC) -> SparseMatrixCSC
    selinv(F::KLU.KLUFactorization, B::SparseMatrixCSC) -> SparseMatrixCSC

Compute the entries of `A⁻¹` at the stored positions of `B`, where `F` is a
sparse LU factorization of a square, nonsingular matrix `A`: `F = lu(A)`
(UMFPACK), or `F = klu(A)` (KLU, available after `using KLU`).

The result has exactly the sparsity pattern of `B` (including explicitly stored
zeros of `B`, whose values are ignored) and is indexed like `A`: entry `(i, j)`
of the result is `inv(A)[i, j]`. The factorization's row scaling and its
independent row and column permutations are undone internally, as is KLU's
block triangular form. `A` may be unsymmetric or indefinite, have zeros on its
diagonal, and be real or complex.

Only the entries of `A⁻¹` needed to obtain the requested ones are computed, using
the sparse factors: no dense inverse is formed and no solve is performed per
requested entry. The entries needed are the closure of the requested pattern
under the selected-inversion recurrences; for a request that lies in the
transposed pattern of `A` (for symmetric `A`, the pattern of `A` itself), this
is the transposed pattern of `L + U`.

To repeat the computation for refactorizations of matrices with the same
pattern, prepare an [`LUSelInvWorkspace`](@ref) once and call
[`selinv!`](@ref).

# Example
```julia
F = lu(A)
Z = selinv(F, A)           # inv(A) at the pattern of A
Z = selinv(F, sparse(A'))  # inv(A) at the pattern of A', as in tr(inv(A) * dA)
```
"""
selinv(F::_UmfpackLU, B::SparseMatrixCSC) = _selinv_lu(F, B)

function _selinv_lu(F::Factorization, B::SparseMatrixCSC)
    ws = LUSelInvWorkspace(F, B)
    _selinv_lu_numeric!(ws)
    dest = SparseMatrixCSC(
        size(B, 1), size(B, 2), copy(ws.B_colptr), copy(ws.B_rowval),
        Vector{eltype(ws.X_val)}(undef, length(ws.B_rowval)),
    )
    return _selinv_lu_extract!(dest, ws)
end

"""
    selinv!(dest::SparseMatrixCSC, ws::LUSelInvWorkspace, F) -> dest

Overwrite the stored values of `dest` with the entries of `inv(A)` at the
requested pattern of `ws`, where `F` is a sparse LU factorization of `A`
(`lu(A)`, or `klu(A)` after `using KLU`). `dest` must have the requested
pattern (same `colptr` and `rowval`), for example `similar(B, eltype(F))` for
the `B` passed to [`LUSelInvWorkspace`](@ref).

`F` may be a refactorization (`lu!`, `klu!`) of a different matrix with the same
size. The factorization may choose different pivots for different values.
`selinv!` compares the pivot sequence and factor pattern of `F` with those `ws`
was prepared for. When they agree, only the numeric phase runs; for UMFPACK the
only memory allocated is the temporary `O(n)` workspace UMFPACK uses while
handing out its factors, while reading KLU's factors allocates copies of them.
When they differ, `ws` is prepared again for `F` (allocating) before the
numeric phase. `ws.preparations` counts the preparations, including the
initial one.
"""
selinv!(dest::SparseMatrixCSC, ws::LUSelInvWorkspace, F::_UmfpackLU) = _selinv_lu!(dest, ws, F)

function _selinv_lu!(dest::SparseMatrixCSC, ws::LUSelInvWorkspace, F::Factorization)
    size(F) == size(ws) || throw(
        DimensionMismatch("factorization has size $(size(F)), workspace has size $(size(ws))"),
    )
    _check_dest_pattern(dest, ws)
    _read_factors!(ws.factors, F)
    _factor_structure_matches(ws) || _prepare!(ws)
    _selinv_lu_numeric!(ws)
    return _selinv_lu_extract!(dest, ws)
end

"""
    selinv_diag(F::SparseArrays.UMFPACK.UmfpackLU; depermute=true) -> Vector
    selinv_diag(F::KLU.KLUFactorization; depermute=true) -> Vector

Compute the diagonal of `inv(A)` from a sparse LU factorization `F` of `A`
(`lu(A)`, or `klu(A)` after `using KLU`). Entry `i` of the result is
`inv(A)[i, i]` in the original numbering of `A`; `depermute=false` is not
supported because the row and column permutations of a pivoted LU factorization
differ.
"""
selinv_diag(F::_UmfpackLU; depermute = true) = _selinv_lu_diag(F, depermute)

function _selinv_lu_diag(F::Factorization, depermute::Bool)
    depermute || throw(
        ArgumentError("LU selected inverses are only available in the original numbering (depermute=true)"),
    )
    n = _check_lu_factorization(F)
    D = SparseMatrixCSC(n, n, collect(1:(n + 1)), collect(1:n), ones(Bool, n))
    return nonzeros(_selinv_lu(F, D))
end

function _check_dest_pattern(dest::SparseMatrixCSC, ws::LUSelInvWorkspace)
    size(dest) == size(ws) || throw(
        DimensionMismatch("dest has size $(size(dest)), workspace has size $(size(ws))"),
    )
    (SparseArrays.getcolptr(dest) == ws.B_colptr && rowvals(dest) == ws.B_rowval) ||
        throw(ArgumentError("dest must have the sparsity pattern the workspace was prepared for"))
    return nothing
end

function _factor_structure_matches(ws::LUSelInvWorkspace)
    f = ws.factors
    return f.Lp == ws.prepared_Lp && f.Lj == ws.prepared_Lj &&
        f.Up == ws.prepared_Up && f.Ui == ws.prepared_Ui &&
        f.P == ws.prepared_P && f.Q == ws.prepared_Q
end

# ── Symbolic preparation ─────────────────────────────────────────────────────

function _prepare!(ws::LUSelInvWorkspace{Tv}) where {Tv}
    f = ws.factors
    n = ws.n
    for (dst, src) in (
            (ws.prepared_Lp, f.Lp), (ws.prepared_Lj, f.Lj), (ws.prepared_Up, f.Up),
            (ws.prepared_Ui, f.Ui), (ws.prepared_P, f.P), (ws.prepared_Q, f.Q),
        )
        resize!(dst, length(src))
        copyto!(dst, src)
    end
    _prepare_L!(ws)
    _prepare_U!(ws)
    pinv = Vector{Int}(undef, n)
    qinv = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        pinv[f.P[k] + 1] = k
        qinv[f.Q[k] + 1] = k
    end
    req_lo_ptr, req_lo_idx, req_up_ptr, req_up_idx = _requested_entries(ws, pinv, qinv)
    _closure!(ws, req_lo_ptr, req_lo_idx, req_up_ptr, req_up_idx)
    _request_sources!(ws, pinv, qinv)
    resize!(ws.X_val, length(ws.lo_rowval) + length(ws.up_colval) + n)
    ws.preparations += 1
    return ws
end

# Strictly lower part of L, by columns, from UMFPACK's row storage.
function _prepare_L!(ws::LUSelInvWorkspace{Tv}) where {Tv}
    f = ws.factors
    n = ws.n
    colptr = resize!(ws.L_colptr, n + 1)
    fill!(colptr, 0)
    @inbounds for i in 1:n, t in (f.Lp[i] + 1):f.Lp[i + 1]
        k = f.Lj[t] + 1
        k < i && (colptr[k + 1] += 1)
    end
    colptr[1] = 1
    @inbounds for k in 1:n
        colptr[k + 1] += colptr[k]
    end
    nz = colptr[n + 1] - 1
    rowval = resize!(ws.L_rowval, nz)
    src = resize!(ws.L_src, nz)
    resize!(ws.L_val, nz)
    next = colptr[1:n]
    # Rows are visited in increasing order, so each column comes out sorted.
    @inbounds for i in 1:n, t in (f.Lp[i] + 1):f.Lp[i + 1]
        k = f.Lj[t] + 1
        if k < i
            s = next[k]
            rowval[s] = i
            src[s] = t
            next[k] = s + 1
        end
    end
    return ws
end

# Strictly upper part of U, by rows, from UMFPACK's column storage.
function _prepare_U!(ws::LUSelInvWorkspace{Tv}) where {Tv}
    f = ws.factors
    n = ws.n
    rowptr = resize!(ws.U_rowptr, n + 1)
    fill!(rowptr, 0)
    diag_src = resize!(ws.U_diag_src, n)
    fill!(diag_src, 0)
    resize!(ws.U_diag, n)
    @inbounds for j in 1:n, t in (f.Up[j] + 1):f.Up[j + 1]
        k = f.Ui[t] + 1
        if k < j
            rowptr[k + 1] += 1
        elseif k == j
            diag_src[j] = t
        end
    end
    rowptr[1] = 1
    @inbounds for k in 1:n
        rowptr[k + 1] += rowptr[k]
    end
    nz = rowptr[n + 1] - 1
    colval = resize!(ws.U_colval, nz)
    src = resize!(ws.U_src, nz)
    resize!(ws.U_val, nz)
    next = rowptr[1:n]
    # Columns are visited in increasing order, so each row comes out sorted.
    @inbounds for j in 1:n, t in (f.Up[j] + 1):f.Up[j + 1]
        k = f.Ui[t] + 1
        if k < j
            s = next[k]
            colval[s] = j
            src[s] = t
            next[k] = s + 1
        end
    end
    return ws
end

# Requested entries of A⁻¹, as entries (r, c) of X: those below the diagonal
# grouped by column c, those above grouped by row r. Diagonal entries of X are
# always computed and need no record.
function _requested_entries(ws::LUSelInvWorkspace, pinv, qinv)
    n = ws.n
    lo_ptr = zeros(Int, n + 1)
    up_ptr = zeros(Int, n + 1)
    B_colptr, B_rowval = ws.B_colptr, ws.B_rowval
    @inbounds for b in 1:n, t in B_colptr[b]:(B_colptr[b + 1] - 1)
        r, c = qinv[B_rowval[t]], pinv[b]
        if r > c
            lo_ptr[c + 1] += 1
        elseif r < c
            up_ptr[r + 1] += 1
        end
    end
    lo_ptr[1] = up_ptr[1] = 1
    @inbounds for k in 1:n
        lo_ptr[k + 1] += lo_ptr[k]
        up_ptr[k + 1] += up_ptr[k]
    end
    lo_idx = Vector{Int}(undef, lo_ptr[n + 1] - 1)
    up_idx = Vector{Int}(undef, up_ptr[n + 1] - 1)
    lo_next = lo_ptr[1:n]
    up_next = up_ptr[1:n]
    @inbounds for b in 1:n, t in B_colptr[b]:(B_colptr[b + 1] - 1)
        r, c = qinv[B_rowval[t]], pinv[b]
        if r > c
            lo_idx[lo_next[c]] = r
            lo_next[c] += 1
        elseif r < c
            up_idx[up_next[r]] = c
            up_next[r] += 1
        end
    end
    return lo_ptr, lo_idx, up_ptr, up_idx
end

# First index s in lo:hi with v[s] > x (hi + 1 if none); v[lo:hi] is sorted.
@inline function _first_greater(v, lo, hi, x)
    return searchsortedfirst(v, x + 1, lo, hi, Base.Order.Forward)
end

# Compute the closure N of the requested entries and the diagonal (see the
# header comment) column by column. With low(i) the rows r > i of column i of N
# and upp(i) the columns c > i of row i of N, the closure rules give, for
# ascending i,
#
#   low(i) = req ∪ R(i) ∪ ⋃_{k<i, i ∈ C(k)} low(k) ∩ (i, n]
#                       ∪ ⋃_{k<i, i ∈ upp(k)} R(k) ∩ (i, n]
#   upp(i) = req ∪ ⋃_{k<i, i ∈ low(k)} C(k) ∩ (i, n]
#                ∪ ⋃_{k<i, i ∈ R(k)} upp(k) ∩ (i, n]
#
# Every set on the right-hand sides is final when i is reached. The `k` with
# i ∈ C(k) (row i of L) and i ∈ R(k) (column i of U) are read from UMFPACK's
# native storage; the `k` with i ∈ low(k) or i ∈ upp(k) are kept in linked lists.
function _closure!(ws::LUSelInvWorkspace, req_lo_ptr, req_lo_idx, req_up_ptr, req_up_idx)
    f = ws.factors
    n = ws.n
    L_colptr, L_rowval = ws.L_colptr, ws.L_rowval
    U_rowptr, U_colval = ws.U_rowptr, ws.U_colval
    lo_colptr = resize!(ws.lo_colptr, n + 1)
    up_rowptr = resize!(ws.up_rowptr, n + 1)
    lo_rowval = empty!(ws.lo_rowval)
    up_colval = empty!(ws.up_colval)
    sizehint!(lo_rowval, length(U_colval) + length(req_lo_idx))
    sizehint!(up_colval, length(L_rowval) + length(req_up_idx))
    lo_colptr[1] = 1
    up_rowptr[1] = 1
    seen_lo = zeros(Int, n)
    seen_up = zeros(Int, n)
    # Linked lists: lo_head[r] → columns k < r with r ∈ low(k); up_head[c] → rows
    # k < c with c ∈ upp(k).
    lo_head = zeros(Int, n)
    up_head = zeros(Int, n)
    lo_list_k = Int[]
    lo_list_next = Int[]
    up_list_k = Int[]
    up_list_next = Int[]
    @inbounds for i in 1:n
        # ── low(i)
        for t in req_lo_ptr[i]:(req_lo_ptr[i + 1] - 1)
            r = req_lo_idx[t]
            seen_lo[r] == i || (seen_lo[r] = i; push!(lo_rowval, r))
        end
        for t in U_rowptr[i]:(U_rowptr[i + 1] - 1)
            r = U_colval[t]
            seen_lo[r] == i || (seen_lo[r] = i; push!(lo_rowval, r))
        end
        for t in (f.Lp[i] + 1):f.Lp[i + 1]          # row i of L
            k = f.Lj[t] + 1
            k < i || continue
            hi = lo_colptr[k + 1] - 1
            for s in _first_greater(lo_rowval, lo_colptr[k], hi, i):hi
                r = lo_rowval[s]
                seen_lo[r] == i || (seen_lo[r] = i; push!(lo_rowval, r))
            end
        end
        node = up_head[i]                           # k < i with i ∈ upp(k)
        while node != 0
            k = up_list_k[node]
            hi = U_rowptr[k + 1] - 1
            for s in _first_greater(U_colval, U_rowptr[k], hi, i):hi
                r = U_colval[s]
                seen_lo[r] == i || (seen_lo[r] = i; push!(lo_rowval, r))
            end
            node = up_list_next[node]
        end
        sort!(view(lo_rowval, lo_colptr[i]:length(lo_rowval)))
        lo_colptr[i + 1] = length(lo_rowval) + 1
        # ── upp(i)
        for t in req_up_ptr[i]:(req_up_ptr[i + 1] - 1)
            c = req_up_idx[t]
            seen_up[c] == i || (seen_up[c] = i; push!(up_colval, c))
        end
        node = lo_head[i]                           # k < i with i ∈ low(k)
        while node != 0
            k = lo_list_k[node]
            hi = L_colptr[k + 1] - 1
            for s in _first_greater(L_rowval, L_colptr[k], hi, i):hi
                c = L_rowval[s]
                seen_up[c] == i || (seen_up[c] = i; push!(up_colval, c))
            end
            node = lo_list_next[node]
        end
        for t in (f.Up[i] + 1):f.Up[i + 1]          # column i of U
            k = f.Ui[t] + 1
            k < i || continue
            hi = up_rowptr[k + 1] - 1
            for s in _first_greater(up_colval, up_rowptr[k], hi, i):hi
                c = up_colval[s]
                seen_up[c] == i || (seen_up[c] = i; push!(up_colval, c))
            end
        end
        sort!(view(up_colval, up_rowptr[i]:length(up_colval)))
        up_rowptr[i + 1] = length(up_colval) + 1
        # ── record column i of low and row i of upp in the linked lists
        for s in lo_colptr[i]:(lo_colptr[i + 1] - 1)
            r = lo_rowval[s]
            push!(lo_list_k, i)
            push!(lo_list_next, lo_head[r])
            lo_head[r] = length(lo_list_k)
        end
        for s in up_rowptr[i]:(up_rowptr[i + 1] - 1)
            c = up_colval[s]
            push!(up_list_k, i)
            push!(up_list_next, up_head[c])
            up_head[c] = length(up_list_k)
        end
    end
    return ws
end

# For each requested entry of A⁻¹, its source in X_val.
function _request_sources!(ws::LUSelInvWorkspace, pinv, qinv)
    n = ws.n
    nlo = length(ws.lo_rowval)
    nup = length(ws.up_colval)
    B_colptr, B_rowval = ws.B_colptr, ws.B_rowval
    src = resize!(ws.B_src, length(B_rowval))
    @inbounds for b in 1:n, t in B_colptr[b]:(B_colptr[b + 1] - 1)
        r, c = qinv[B_rowval[t]], pinv[b]
        src[t] = if r > c
            _find_sorted(ws.lo_rowval, ws.lo_colptr[c], ws.lo_colptr[c + 1] - 1, r)
        elseif r < c
            nlo + _find_sorted(ws.up_colval, ws.up_rowptr[r], ws.up_rowptr[r + 1] - 1, c)
        else
            nlo + nup + r
        end
    end
    return ws
end

@inline function _find_sorted(v, lo, hi, x)
    s = searchsortedfirst(v, x, lo, hi, Base.Order.Forward)
    (s <= hi && v[s] == x) || _closure_error()
    return s
end

@noinline _closure_error() = error(
    "SelectedInversion: an entry required by the LU selected-inversion recurrence is " *
        "missing from the prepared structure (internal error)",
)

# ── Numeric phase ────────────────────────────────────────────────────────────

# First index s ≥ lo in the sorted v[lo:hi] with v[s] ≥ x (hi + 1 if none).
# Exponential search keeps short skips cheap.
@inline function _gallop_geq(v, lo, hi, x)
    @inbounds begin
        (lo > hi || v[lo] >= x) && return lo
        step = 1
        while lo + step <= hi && v[lo + step] < x
            lo += step
            step <<= 1
        end
        return searchsortedfirst(v, x, lo + 1, min(lo + step, hi), Base.Order.Forward)
    end
end

# Position of x in the sorted v[lo:hi], searching forward from lo.
@inline function _gallop_find(v, lo, hi, x)
    s = _gallop_geq(v, lo, hi, x)
    @inbounds (s <= hi && v[s] == x) || _closure_error()
    return s
end

function _selinv_lu_numeric!(ws::LUSelInvWorkspace{Tv}) where {Tv}
    f = ws.factors
    n = ws.n
    L_colptr, L_rowval, L_val = ws.L_colptr, ws.L_rowval, ws.L_val
    U_rowptr, U_colval, U_val = ws.U_rowptr, ws.U_colval, ws.U_val
    U_diag = ws.U_diag
    @inbounds begin
        for t in eachindex(L_val)
            L_val[t] = f.Lx[ws.L_src[t]]
        end
        for t in eachindex(U_val)
            U_val[t] = f.Ux[ws.U_src[t]]
        end
        for k in 1:n
            s = ws.U_diag_src[k]
            U_diag[k] = iszero(s) ? zero(Tv) : f.Ux[s]
        end
    end
    _row_multipliers!(ws.row_multipliers, f)
    lo_colptr, lo_rowval = ws.lo_colptr, ws.lo_rowval
    up_rowptr, up_colval = ws.up_rowptr, ws.up_colval
    X = ws.X_val
    nlo = length(lo_rowval)
    nup = length(up_colval)
    dg = nlo + nup
    @inbounds for k in n:-1:1
        ukk = U_diag[k]
        iszero(ukk) && throw(SingularException(Int(f.Q[k]) + 1))
        inv_ukk = inv(ukk)
        a_lo, a_hi = lo_colptr[k], lo_colptr[k + 1] - 1      # low(k)
        l_lo, l_hi = L_colptr[k], L_colptr[k + 1] - 1        # C(k)
        c_lo, c_hi = up_rowptr[k], up_rowptr[k + 1] - 1      # upp(k)
        u_lo, u_hi = U_rowptr[k], U_rowptr[k + 1] - 1        # R(k)

        # ── X[r, k] = -Σ_{i ∈ C(k)} X[r, i] L[i, k] for r ∈ low(k)
        for a in a_lo:a_hi
            X[a] = zero(Tv)
        end
        # Entries X[r, i] with r ≥ i: diagonal or column i of the lower part.
        a0 = a_lo
        for s in l_lo:l_hi
            i = L_rowval[s]
            lik = L_val[s]
            while a0 <= a_hi && lo_rowval[a0] < i
                a0 += 1
            end
            a = a0
            if a <= a_hi && lo_rowval[a] == i
                X[a] += X[dg + i] * lik
                a += 1
            end
            b = lo_colptr[i]
            b_hi = lo_colptr[i + 1] - 1
            while a <= a_hi
                b = _gallop_find(lo_rowval, b, b_hi, lo_rowval[a])
                X[a] += X[b] * lik
                a += 1
                b += 1
            end
        end
        # Entries X[r, i] with r < i: row r of the upper part.
        s0 = l_lo
        for a in a_lo:a_hi
            r = lo_rowval[a]
            while s0 <= l_hi && L_rowval[s0] <= r
                s0 += 1
            end
            s0 > l_hi && break
            b = up_rowptr[r]
            b_hi = up_rowptr[r + 1] - 1
            acc = zero(Tv)
            for s in s0:l_hi
                b = _gallop_find(up_colval, b, b_hi, L_rowval[s])
                acc += X[nlo + b] * L_val[s]
                b += 1
            end
            X[a] += acc
        end
        for a in a_lo:a_hi
            X[a] = -X[a]
        end

        # ── X[k, c] = -Σ_{j ∈ R(k)} U[k, j] X[j, c] / U[k, k] for c ∈ upp(k)
        for c in c_lo:c_hi
            X[nlo + c] = zero(Tv)
        end
        # Entries X[j, c] with j ≤ c: diagonal or row j of the upper part.
        c0 = c_lo
        for s in u_lo:u_hi
            j = U_colval[s]
            ukj = U_val[s]
            while c0 <= c_hi && up_colval[c0] < j
                c0 += 1
            end
            c = c0
            if c <= c_hi && up_colval[c] == j
                X[nlo + c] += ukj * X[dg + j]
                c += 1
            end
            b = up_rowptr[j]
            b_hi = up_rowptr[j + 1] - 1
            while c <= c_hi
                b = _gallop_find(up_colval, b, b_hi, up_colval[c])
                X[nlo + c] += ukj * X[nlo + b]
                c += 1
                b += 1
            end
        end
        # Entries X[j, c] with j > c: column c of the lower part.
        s0 = u_lo
        for c in c_lo:c_hi
            col = up_colval[c]
            while s0 <= u_hi && U_colval[s0] <= col
                s0 += 1
            end
            s0 > u_hi && break
            b = lo_colptr[col]
            b_hi = lo_colptr[col + 1] - 1
            acc = zero(Tv)
            for s in s0:u_hi
                b = _gallop_find(lo_rowval, b, b_hi, U_colval[s])
                acc += U_val[s] * X[b]
                b += 1
            end
            X[nlo + c] += acc
        end
        for c in c_lo:c_hi
            X[nlo + c] = -X[nlo + c] * inv_ukk
        end

        # ── X[k, k] = (1 - Σ_{j ∈ R(k)} U[k, j] X[j, k]) / U[k, k]
        acc = zero(Tv)
        b = a_lo
        for s in u_lo:u_hi
            b = _gallop_find(lo_rowval, b, a_hi, U_colval[s])
            acc += U_val[s] * X[b]
            b += 1
        end
        X[dg + k] = (one(Tv) - acc) * inv_ukk
    end
    return ws
end

# inv(A)[a, b] = X[qinv[a], pinv[b]] * r[b]
function _selinv_lu_extract!(dest::SparseMatrixCSC, ws::LUSelInvWorkspace)
    nz = nonzeros(dest)
    X = ws.X_val
    src = ws.B_src
    rm = ws.row_multipliers
    B_colptr = ws.B_colptr
    @inbounds for b in 1:ws.n
        rb = rm[b]
        for t in B_colptr[b]:(B_colptr[b + 1] - 1)
            nz[t] = X[src[t]] * rb
        end
    end
    return dest
end
