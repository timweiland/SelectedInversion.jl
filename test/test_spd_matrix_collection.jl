using SelectedInversion

using SuiteSparseMatrixCollection
using MatrixMarket
using LinearAlgebra, SparseArrays

MAX_ROWS = 1000
# The following matrices are particularly ill-conditioned, so comparing to a
# naive inverse fails (we actually expect SelInv to be more "correct"/stable here)
EXCLUDE_MATS = ["plat362"]

@testset "SPD SuiteSparse matrices" begin
    ssmc = ssmc_db()
    SPD_mats_tiny = ssmc[
        (ssmc.numerical_symmetry.==1).&(ssmc.positive_definite.==true).&(ssmc.real.==true).&(ssmc.nrows.≤MAX_ROWS).&(ssmc.name.∉Ref(
            EXCLUDE_MATS,
        )),
        :,
    ]
    paths = fetch_ssmc(SPD_mats_tiny, format = "MM")
    paths =
        [joinpath(path, "$(SPD_mats_tiny.name[i]).mtx") for (i, path) in enumerate(paths)]

    has_simplicial = false
    has_supernodal = false

    for path in paths
        A = MatrixMarket.mmread(path)
        A⁻¹ = inv(Array(A))
        C = cholesky(A)

        if Bool(unsafe_load(pointer(C)).is_super)
            has_supernodal = true
        else
            has_simplicial = true
        end

        Z = selinv(C; depermute = true)[1]
        @test check_selinv(Z, A⁻¹)
    end

    @test has_simplicial
    @test has_supernodal
end
