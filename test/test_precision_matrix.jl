using SelectedInversion

using JLD
using LinearAlgebra, SparseArrays
using Random

@testset "Precision matrix of a GMRF" begin
    Q = JLD.load("data/Q_gmrf.jld")["Q"]
    N = size(Q, 2)

    Q_cho = cholesky(Q)
    Z = selinv(Q_cho; depermute=true).Z
    d = diag(Z)

    # The diagonal corresponds to the marginal variances, which must be positive
    @test all(d .> 0.)

    rng = MersenneTwister(9453778)
    N_test = 30
    test_idcs = shuffle(rng, 1:N)[1:N_test]

    for idx in test_idcs
        e = zeros(N)
        e[idx] = 1.
        ground_truth = dot(e, Q_cho \ e)
        @test d[idx] ≈ ground_truth atol=1e-7
    end
end
