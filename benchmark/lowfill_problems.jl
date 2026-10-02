# Low-fill SPD test problems from latent Gaussian models. Their Cholesky factors
# have only a few nonzeros per column, so CHOLMOD factorizes them with the
# simplicial (non-supernodal) method. A 2D grid serves as a supernodal control.

using SparseArrays, LinearAlgebra, Random

"""
    tree_gmrf(n_leaves; n_fields=1, rng) -> SparseMatrixCSC

Posterior precision of a Gaussian field on a random rooted binary tree with
`n_leaves` observed leaves. Each node is its parent plus independent noise; the
root is fixed, leaving `2n_leaves - 2` latent nodes. With `n_fields > 1`,
correlated fields share the tree, with a random `n_fields × n_fields` covariance.
"""
function tree_gmrf(n_leaves::Int; n_fields::Int = 1, rng = MersenneTwister(1))
    n_nodes = 2n_leaves - 1
    parent = zeros(Int, n_nodes)
    roots = collect(1:n_leaves)
    for node in (n_leaves + 1):n_nodes
        i, j = randperm(rng, length(roots))[1:2]
        parent[roots[i]] = parent[roots[j]] = node
        deleteat!(roots, sort([i, j]))
        push!(roots, node)
    end
    n = n_nodes - 1  # the root is node n_nodes
    rows, cols, vals = Int[], Int[], Float64[]
    for c in 1:n
        w = 1 / (0.1 + rand(rng))  # inverse edge variance
        push!(rows, c); push!(cols, c); push!(vals, w)
        par = parent[c]
        if par <= n
            append!(rows, (par, par, c)); append!(cols, (par, c, par)); append!(vals, (w, -w, -w))
        end
    end
    Q = sparse(rows, cols, vals, n, n) + sparse(1:n_leaves, 1:n_leaves, fill(4.0, n_leaves), n, n)
    n_fields == 1 && return Q
    B = randn(rng, n_fields, n_fields)
    return kron(Q, sparse(Symmetric(inv(B * B' + n_fields * I))))
end

"""
    nested_effects(n_groups; group_size=20, n_covariates=10, rng) -> SparseMatrixCSC

Posterior precision of a two-level hierarchical model: each unit's effect is
centered on its group's effect, every unit has one observation, and a few shared
covariate effects enter every observation (dense rows and columns).
"""
function nested_effects(
        n_groups::Int; group_size::Int = 20, n_covariates::Int = 10,
        rng = MersenneTwister(2),
    )
    n_units = n_groups * group_size
    group = repeat(1:n_groups; inner = group_size)
    # Prior precision of (group effects, unit effects)
    D = [-sparse(1:n_units, group, ones(n_units), n_units, n_groups) I]
    P = D'D / 0.5 + blockdiag(sparse(I, n_groups, n_groups), spzeros(n_units, n_units))
    # One observation per unit, depending on its unit effect and the covariates
    X = sprandn(rng, n_units, n_covariates, 0.5)
    U = [spzeros(n_units, n_groups) I]
    H = [X U]' * [X U] / 0.3
    return sparse(Symmetric(H + blockdiag(1.0e-2 * sparse(I, n_covariates, n_covariates), P)))
end

"""
    rw2_gmrf(n; τ=1.0, σ²=0.5) -> SparseMatrixCSC

Posterior precision of a second-order random walk observed with Gaussian noise
(pentadiagonal).
"""
function rw2_gmrf(n::Int; τ = 1.0, σ² = 0.5)
    D = spdiagm(n - 2, n, 0 => ones(n - 2), 1 => -2ones(n - 2), 2 => ones(n - 2))
    return sparse(τ * (D'D) + I / σ²)
end

"""
    grid_gmrf(m) -> SparseMatrixCSC

Shifted 2D Laplacian on an `m × m` grid. Its factor has dense supernodes.
"""
function grid_gmrf(m::Int)
    T = spdiagm(-1 => -ones(m - 1), 0 => 2ones(m), 1 => -ones(m - 1))
    return kron(T, sparse(I, m, m)) + kron(sparse(I, m, m), T) + 0.1I
end

lowfill_problems() = [
    "tree_1k" => () -> tree_gmrf(1_000),
    "tree_10k" => () -> tree_gmrf(10_000),
    "tree_100k" => () -> tree_gmrf(100_000),
    "tree_10k_3f" => () -> tree_gmrf(10_000; n_fields = 3),
    "nested_100" => () -> nested_effects(100),
    "nested_1k" => () -> nested_effects(1_000),
    "nested_5k" => () -> nested_effects(5_000),
    "rw2_100k" => () -> rw2_gmrf(100_000),
    "rw2_1M" => () -> rw2_gmrf(1_000_000),
    "grid_100" => () -> grid_gmrf(100),
    "grid_300" => () -> grid_gmrf(300),
]
