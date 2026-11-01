# Numeric factors of a pivoted sparse LU factorization in one internal layout,
# shared by every supported factorization (UMFPACK in `umfpack_factors.jl`, KLU
# through the `SelectedInversionKLUExt` extension).

"""
Numeric factors of a square factorization `(R A)[P, Q] = L U`, in UMFPACK's
native 0-based layout: `L` unit lower triangular by rows and `U` upper
triangular by columns, each possibly storing its diagonal. `R` is the row
scaling, indexed by the rows of `A`: multiplication by `Rs` when `do_recip` is
nonzero and division otherwise. The buffers are reused across reads.
"""
mutable struct _LUFactors{Tv, Ti}
    n::Int
    Lp::Vector{Ti}
    Lj::Vector{Ti}
    Lx::Vector{Tv}
    Up::Vector{Ti}
    Ui::Vector{Ti}
    Ux::Vector{Tv}
    P::Vector{Ti}
    Q::Vector{Ti}
    Rs::Vector{Float64}
    do_recip::Base.RefValue{Ti}
    lnz::Base.RefValue{Ti}
    unz::Base.RefValue{Ti}
    nrow::Base.RefValue{Ti}
    ncol::Base.RefValue{Ti}
    nzud::Base.RefValue{Ti}
end

function _LUFactors{Tv, Ti}() where {Tv, Ti}
    return _LUFactors{Tv, Ti}(
        0, Ti[], Ti[], Tv[], Ti[], Ti[], Tv[], Ti[], Ti[], Float64[],
        Ref(zero(Ti)), Ref(zero(Ti)), Ref(zero(Ti)), Ref(zero(Ti)), Ref(zero(Ti)),
        Ref(zero(Ti)),
    )
end

function _check_lu_factorization(F::Factorization)
    m, n = size(F)
    m == n || throw(DimensionMismatch("selected inversion requires a square matrix, got $m×$n"))
    issuccess(F) || throw(SingularException(0))
    return n
end

"""
    _row_multipliers!(r, f)

Write the factors by which the factorization multiplied each row of `A`, i.e.
`r[i]` such that `(r .* A)[P, Q] = L U`.
"""
function _row_multipliers!(r::Vector{Float64}, f::_LUFactors)
    resize!(r, f.n)
    if iszero(f.do_recip[])
        @inbounds for i in 1:f.n
            r[i] = inv(f.Rs[i])
        end
    else
        copyto!(r, f.Rs)
    end
    return r
end

function _copy_zero_based!(dst::Vector{Ti}, src::AbstractVector) where {Ti}
    resize!(dst, length(src))
    @inbounds for t in eachindex(dst, src)
        dst[t] = Ti(src[t] - 1)
    end
    return dst
end

"""
    _unit_lower_solve(L, F) -> SparseMatrixCSC

Solve `L G = F` for a unit lower triangular sparse `L` (its stored diagonal, if
any, is ignored) and a sparse right-hand side `F`. Column `j` of `G` is stored on
the set of rows reachable from the rows of `F[:, j]` in the graph of `L`
(Gilbert & Peierls, 1988), so the cost is proportional to the work of the solve.
"""
function _unit_lower_solve(L::SparseMatrixCSC{Tv}, F::SparseMatrixCSC{Tf}) where {Tv, Tf}
    T = promote_type(Tv, Tf)
    n = size(L, 1)
    m = size(F, 2)
    Lp, Li, Lx = L.colptr, rowvals(L), nonzeros(L)
    Fp, Fi, Fx = F.colptr, rowvals(F), nonzeros(F)
    colptr = Vector{Int}(undef, m + 1)
    colptr[1] = 1
    rowval = Int[]
    nzval = T[]
    x = zeros(T, n)
    mark = zeros(Int, n)
    stack = Vector{Int}(undef, n)
    next = Vector{Int}(undef, n)
    order = Vector{Int}(undef, n)
    @inbounds for j in 1:m
        # Depth-first search from each row of F[:, j]; vertices are appended to
        # `order` from its end in postorder, so order[top:n] is topological.
        top = n + 1
        for t in Fp[j]:(Fp[j + 1] - 1)
            root = Fi[t]
            mark[root] == j && continue
            mark[root] = j
            depth = 1
            stack[1] = root
            next[1] = Lp[root]
            while depth > 0
                i = stack[depth]
                s = next[depth]
                descended = false
                while s < Lp[i + 1]
                    k = Li[s]
                    s += 1
                    if k > i && mark[k] != j
                        mark[k] = j
                        next[depth] = s
                        depth += 1
                        stack[depth] = k
                        next[depth] = Lp[k]
                        descended = true
                        break
                    end
                end
                if !descended
                    depth -= 1
                    top -= 1
                    order[top] = i
                end
            end
        end
        for t in Fp[j]:(Fp[j + 1] - 1)
            x[Fi[t]] = Fx[t]
        end
        for o in top:n
            i = order[o]
            xi = x[i]
            iszero(xi) && continue
            for s in Lp[i]:(Lp[i + 1] - 1)
                k = Li[s]
                k > i && (x[k] -= Lx[s] * xi)
            end
        end
        rows = sort!(view(order, top:n))
        for i in rows
            push!(rowval, i)
            push!(nzval, x[i])
            x[i] = zero(T)
        end
        colptr[j + 1] = length(rowval) + 1
    end
    return SparseMatrixCSC(n, m, colptr, rowval, nzval)
end

"""
    _set_factors!(f, L, U, F, p, q, Rs)

Load factors of a block-triangular factorization
`L * U + F == Diagonal(1 ./ Rs) * A[p, q]` (the KLU convention: `L` and `U` are
block diagonal, `F` holds the strictly block-upper part, and `Rs` divides the
rows in permuted order) into `f`. Since `L` is block diagonal,
`L * U + F == L * (U + L⁻¹F)` is the factorization of the permuted, scaled
matrix without further pivoting, so `U + L⁻¹F` is stored as the upper factor.
"""
function _set_factors!(
        f::_LUFactors{Tv, Ti}, L::SparseMatrixCSC, U::SparseMatrixCSC,
        F::SparseMatrixCSC, p::AbstractVector, q::AbstractVector, Rs::AbstractVector,
    ) where {Tv, Ti}
    n = size(L, 1)
    Ug = iszero(nnz(F)) ? U : U + _unit_lower_solve(L, F)
    Lrows = copy(transpose(L))
    f.n = n
    _copy_zero_based!(f.Lp, SparseArrays.getcolptr(Lrows))
    _copy_zero_based!(f.Lj, rowvals(Lrows))
    copyto!(resize!(f.Lx, nnz(Lrows)), nonzeros(Lrows))
    _copy_zero_based!(f.Up, SparseArrays.getcolptr(Ug))
    _copy_zero_based!(f.Ui, rowvals(Ug))
    copyto!(resize!(f.Ux, nnz(Ug)), nonzeros(Ug))
    _copy_zero_based!(f.P, p)
    _copy_zero_based!(f.Q, q)
    resize!(f.Rs, n)
    @inbounds for i in 1:n
        f.Rs[p[i]] = Rs[i]
    end
    f.do_recip[] = zero(Ti)
    return f
end
