using SelectedInversion

using SuiteSparseMatrixCollection
using MatrixMarket
using LinearAlgebra, SparseArrays

MAX_ROWS = 1000

@testset "SPD SuiteSparse matrices" begin
    ssmc = ssmc_db()
    SPD_mats_tiny = ssmc[(ssmc.numerical_symmetry .== 1) .& (ssmc.positive_definite.== true) .&
        (ssmc.real .== true) .& (ssmc.nrows .≤ MAX_ROWS), :]
    paths = fetch_ssmc(SPD_mats_tiny, format="MM")
    paths = [joinpath(path, "$(SPD_mats_tiny.name[i]).mtx") for (i, path) in enumerate(paths)]

    has_simplicial = false
    has_supernodal = false
    N_supernodal = 0

    for path in paths
        A = MatrixMarket.mmread(path)
        A⁻¹ = inv(Array(A))
        C = cholesky(A)

        if Bool(unsafe_load(pointer(C)).is_super)
            has_supernodal = true
        else
            has_simplicial = true
            continue
        end

        Z = selinv(C)
        @test check_supernodal_chunks_equal_dense(Z, A⁻¹[C.p, C.p])

        N_supernodal += 1
    end

    @test has_simplicial
    @test has_supernodal
end
