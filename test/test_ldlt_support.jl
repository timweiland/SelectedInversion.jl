using Test
using LinearAlgebra
using SparseArrays
using SelectedInversion

include("utils.jl")

@testset "LDLT Support" begin
    # Test the exact example from ldlt_support.org
    ρ, k = 0.95, 100  # Using smaller k for testing performance
    A_tri = SymTridiagonal(ones(k) .+ ρ^2, -ρ * ones(k-1))

    @testset "SymTridiagonal Matrix Support" begin
        # These should work after implementation
        @test_nowarn selinv(A_tri)
        @test_nowarn selinv_diag(A_tri)

        # Test consistency with sparse matrix conversion
        A_sparse = sparse(A_tri)
        Z_sparse, p_sparse = selinv(A_sparse)
        Z_tri, p_tri = selinv(A_tri)

        # Results should be equivalent (possibly up to permutation)
        @test size(Z_sparse) == size(Z_tri)
        @test length(p_sparse) == length(p_tri)
    end

    @testset "LDLt Factorization Support" begin
        F_ldlt = ldlt(A_tri)

        # These should work after implementation
        @test_nowarn selinv(F_ldlt)
        @test_nowarn selinv_diag(F_ldlt)

        # Test consistency between matrix and factorization approaches
        Z_matrix, p_matrix = selinv(A_tri)
        Z_factor, p_factor = selinv(F_ldlt)

        @test Z_matrix ≈ Z_factor
        @test p_matrix == p_factor

        # Test diagonal consistency
        d_matrix = selinv_diag(A_tri)
        d_factor = selinv_diag(F_ldlt)

        @test d_matrix ≈ d_factor
    end

    @testset "Permutation Handling" begin
        F_ldlt = ldlt(A_tri)

        # Test depermute=false
        Z_permuted, p = selinv(F_ldlt; depermute = false)
        d_permuted = selinv_diag(F_ldlt; depermute = false)

        # Test depermute=true (default for selinv_diag)
        Z_depermuted, p_dep = selinv(F_ldlt; depermute = true)
        d_depermuted = selinv_diag(F_ldlt; depermute = true)

        @test size(Z_permuted) == size(Z_depermuted)
        @test length(d_permuted) == length(d_depermuted)
    end

    @testset "Correctness Verification" begin
        # Test on smaller matrix for exact verification using check_selinv helper
        ρ_small, k_small = 0.5, 20
        A_small = SymTridiagonal(ones(k_small) .+ ρ_small^2, -ρ_small * ones(k_small-1))

        # Compute ground truth inverse
        A_inv = inv(Matrix(A_small))

        # Test selinv correctness
        Z, p = selinv(A_small; depermute = true)
        @test check_selinv(sparse(Z), A_inv)

        # Test selinv with LDLt factorization
        F_ldlt = ldlt(A_small)
        Z_ldlt, p_ldlt = selinv(F_ldlt; depermute = true)
        @test check_selinv(sparse(Z_ldlt), A_inv)

        # Test selinv_diag correctness
        d_true = diag(A_inv)

        d_tri = selinv_diag(A_small; depermute = true)
        @test d_tri ≈ d_true

        d_ldlt = selinv_diag(F_ldlt; depermute = true)
        @test d_ldlt ≈ d_true

        # Test default behavior (should be same as depermute=true for selinv_diag)
        d_default = selinv_diag(F_ldlt)
        @test d_default ≈ d_true
        @test d_default ≈ d_ldlt
    end
end
