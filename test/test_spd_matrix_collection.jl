using SelectedInversion

using SuiteSparseMatrixCollection
using MatrixMarket
using LinearAlgebra, SparseArrays
using LDLFactorizations

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

    @testset "Matrix $(SPD_mats_tiny.name[i])" for i in eachindex(paths)
        path = paths[i]
        A = MatrixMarket.mmread(path)
        A⁻¹ = inv(Array(A))
        C = cholesky(A)

        is_super = Bool(unsafe_load(pointer(C)).is_super)
        if is_super
            has_supernodal = true
        else
            has_simplicial = true
        end

        Z = selinv(C; depermute = true)[1]
        @test check_selinv(Z, A⁻¹)

        Z_ldlt = selinv(ldlt(A); depermute = true)[1]
        @test check_selinv(Z_ldlt, A⁻¹)

        if A isa SparseMatrixCSC{Int}
            A = Float64.(A)
        end
        Z_ldl = selinv(ldl(A); depermute = true)[1]
        @test check_selinv(Z_ldl, A⁻¹)

        # For simplicial factorizations, check that sparsity pattern is correct
        if !is_super
            GT = sparse(C.L) .!== 0.
            GT = GT + GT'
            p_inv = invperm(C.p)
            GT = GT[p_inv, p_inv]
            check_sparsity_pattern(Z, GT)
            check_sparsity_pattern(Z_ldlt, GT)
            check_sparsity_pattern(Z_ldl, GT)
        end

        # Test selinv_diag correctness
        d_true = diag(A⁻¹)
        
        # Test with depermute=true (should match true diagonal)
        d_selinv_diag = selinv_diag(A; depermute = true)
        @test d_selinv_diag ≈ d_true
        
        # Test default behavior (should be same as depermute=true)
        d_selinv_diag_default = selinv_diag(A)
        @test d_selinv_diag_default ≈ d_true
        @test d_selinv_diag_default ≈ d_selinv_diag
        
        # Test with depermute=false (should match permuted diagonal)
        d_selinv_diag_perm = selinv_diag(A; depermute = false)
        Z_perm = selinv(A; depermute = false)[1]
        @test d_selinv_diag_perm ≈ diag(Z_perm)
        
        # Test with factorization input
        d_from_chol = selinv_diag(C; depermute = true)
        @test d_from_chol ≈ d_true
        
        # Test factorization default behavior
        d_from_chol_default = selinv_diag(C)
        @test d_from_chol_default ≈ d_true
        
        # Test with LDLFactorizations
        d_ldlt = selinv_diag(ldlt(A); depermute = true)
        @test d_ldlt ≈ d_true
        
        d_ldl = selinv_diag(ldl(A); depermute = true)
        @test d_ldl ≈ d_true
    end

    @test has_simplicial
    @test has_supernodal
end
