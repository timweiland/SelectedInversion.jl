using SelectedInversion

using LinearAlgebra, SparseArrays
using Random

# Gaussian field on a random binary tree (root fixed), observed at the leaves.
function simplicial_tree_precision(n_leaves; rng = MersenneTwister(31))
    n_nodes = 2n_leaves - 1
    parent = zeros(Int, n_nodes)
    roots = collect(1:n_leaves)
    for node in (n_leaves + 1):n_nodes
        i, j = randperm(rng, length(roots))[1:2]
        parent[roots[i]] = parent[roots[j]] = node
        deleteat!(roots, sort([i, j]))
        push!(roots, node)
    end
    n = n_nodes - 1
    Q = spzeros(n, n)
    for c in 1:n
        w = 1 / (0.1 + rand(rng))
        Q[c, c] += w
        par = parent[c]
        if par <= n
            Q[par, par] += w
            Q[par, c] -= w
            Q[c, par] -= w
        end
    end
    return Q + sparse(1:n_leaves, 1:n_leaves, fill(4.0, n_leaves), n, n)
end

# Two-level hierarchical model with a few covariates shared by all observations.
function simplicial_nested_precision(n_groups, group_size; rng = MersenneTwister(32))
    n_units = n_groups * group_size
    group = repeat(1:n_groups; inner = group_size)
    D = [-sparse(1:n_units, group, ones(n_units), n_units, n_groups) I]
    P = D'D / 0.5 + blockdiag(sparse(I, n_groups, n_groups), spzeros(n_units, n_units))
    XU = [sprandn(rng, n_units, 4, 0.5) spzeros(n_units, n_groups) I]
    return sparse(Symmetric(XU'XU / 0.3 + blockdiag(1.0e-2 * sparse(I, 4, 4), P)))
end

is_simplicial(F) = !Bool(unsafe_load(pointer(F)).is_super)

function check_simplicial_selinv(F, A)
    A_inv = inv(Matrix(A))
    @test is_simplicial(F)
    Z, p = selinv(F; depermute = true)
    @test check_selinv(Z, A_inv)
    check_sparsity_pattern(Z, A)
    Zp, p2 = selinv(F; depermute = false)
    @test p2 == p
    @test check_selinv(sparse(Zp), A_inv[p, p])
    @test selinv_diag(F) ≈ diag(A_inv)
    @test selinv_diag(F; depermute = false) ≈ diag(A_inv)[p]
    return
end

@testset "Simplicial selected inversion" begin
    @testset "Reject symbolic factors" begin
        A = sparse([2.0 -1 0; -1 2 -1; 0 -1 2])
        F = SparseArrays.CHOLMOD.analyze(SparseArrays.CHOLMOD.Sparse(A))
        @test is_simplicial(F)
        for f in (selinv, selinv_simplicial, selinv_diag)
            @test_throws SparseArrays.CHOLMOD.CHOLMODException("only numeric factors are supported") f(F)
        end
    end

    problems = [
        "tree" => simplicial_tree_precision(150),
        "nested effects" => simplicial_nested_precision(12, 8),
        "tridiagonal" => spdiagm(-1 => -ones(99), 0 => 2.5 * ones(100), 1 => -ones(99)),
        "diagonal" => sparse(Diagonal(1.0:20.0)),
        "1x1" => sparse(fill(4.0, 1, 1)),
    ]
    @testset "$name" for (name, A) in problems
        @testset "LL factor" begin
            check_simplicial_selinv(cholesky(A), A)
        end
        @testset "LDL factor" begin
            check_simplicial_selinv(ldlt(A), A)
        end
    end

    @testset "selinv_simplicial on a supernodal factor" begin
        A = sparse(Symmetric(sprandn(MersenneTwister(33), 300, 300, 0.05) + 30I))
        F = cholesky(A)
        @test !is_simplicial(F)
        Z, p = selinv_simplicial(F; depermute = true)
        @test check_selinv(Z, inv(Matrix(A)))
    end

    @testset "Short columns pointing into a long column" begin
        # Columns 1:K have rows {K + 1, n}; column K + 1 has rows K + 2:n.
        K, M = 30, 40
        n = K + M + 2
        rows = [repeat([K + 1, n], K); (K + 2):n]
        cols = [repeat(1:K; inner = 2); fill(K + 1, M + 1)]
        E = sparse(rows, cols, -0.01, n, n)
        A = E + E' + 2I
        check_simplicial_selinv(cholesky(A; perm = 1:n), A)
    end

    @testset "Unpacked factor after rank update" begin
        A = simplicial_tree_precision(100)
        n = size(A, 1)
        C = sparse([1, 20, 90, n], ones(Int, 4), [1.0, 2.0, -1.0, 0.5], n, 1)
        F = lowrankupdate(cholesky(A), C)
        s = unsafe_load(pointer(F))
        Lp = unsafe_wrap(Array, Ptr{Int}(s.p), n + 1)
        Lnz = unsafe_wrap(Array, Ptr{Int}(s.nz), n)
        # Make sure this exercises a factor with slack between columns
        @test !all(Lp[j + 1] - Lp[j] == Lnz[j] for j in 1:n)
        check_simplicial_selinv(F, A + C * C')
    end
end
