using SelectedInversion
using SparseArrays
using LinearAlgebra
using Test

"""
Create a sparse SPD matrix that produces a supernodal Cholesky factorization.
"""
function make_supernodal_spd(n, density=0.001)
    A = sprand(n, n, density)
    return A * A' + 10I
end

@testset "Parallel Supernodal Selected Inversion" begin

    @testset "Elimination tree construction" begin
        n = 2000
        A = make_supernodal_spd(n, 0.001)
        F = cholesky(A)

        Z = SelectedInversion.SupernodalMatrix(F; transpose_chunks=true, symmetric_access=true)
        etree = SelectedInversion.build_supernodal_etree(Z)

        # Verify tree properties
        @test length(etree.parent) == Z.n_super
        @test etree.parent[Z.n_super] == 0  # Root has no parent
        @test etree.level[Z.n_super] == 0   # Root at level 0
        @test sum(length.(etree.levels)) == Z.n_super  # All supernodes accounted

        # Verify parent-child consistency
        for j in 1:Z.n_super
            for child in etree.children[j]
                @test etree.parent[child] == j
            end
        end

        # Verify levels are properly ordered (children have higher level than parent)
        for j in 1:(Z.n_super - 1)
            if etree.parent[j] > 0
                @test etree.level[j] > etree.level[etree.parent[j]]
            end
        end
    end

    @testset "Parallel vs Sequential correctness" begin
        n = 1000
        A = make_supernodal_spd(n, 0.002)
        F = cholesky(A)

        # Verify we have a supernodal factorization
        s = unsafe_load(pointer(F))
        @test Bool(s.is_super)

        # Run both versions
        Z_seq, p_seq = selinv(F; parallel=false)
        Z_par, p_par = selinv(F; parallel=true)

        # Results should be identical
        @test p_seq == p_par
        @test sparse(Z_seq) ≈ sparse(Z_par)
    end

    @testset "Correctness against true inverse" begin
        n = 200
        A = make_supernodal_spd(n, 0.05)
        F = cholesky(A)

        # Compute actual inverse
        A_inv = inv(Matrix(A))

        # Test parallel selected inversion
        Z_par, p = selinv(F; parallel=true)
        @test check_selinv(Z_par, A_inv)
    end

    @testset "ParallelConfig options" begin
        n = 1000
        A = make_supernodal_spd(n, 0.002)
        F = cholesky(A)

        Z_ref, _ = selinv(F; parallel=false)

        # Test with different min_level_size
        for min_size in [1, 2, 4, 8, 16]
            config = ParallelConfig(true, min_size)
            Z_cfg, _ = selinv(F; parallel=config)
            @test sparse(Z_ref) ≈ sparse(Z_cfg)
        end

        # Test disabled parallel with ParallelConfig
        config_disabled = ParallelConfig(false, 2)
        Z_disabled, _ = selinv(F; parallel=config_disabled)
        @test sparse(Z_ref) ≈ sparse(Z_disabled)
    end

    @testset "Depermute option with parallel" begin
        n = 200
        A = make_supernodal_spd(n, 0.05)
        F = cholesky(A)

        A_inv = inv(Matrix(A))

        # Without depermute - check_selinv applies its own permutation
        Z_perm, p = selinv(F; parallel=true, depermute=false)
        @test check_selinv(Z_perm, A_inv)

        # With depermute
        Z_dep, _ = selinv(F; parallel=true, depermute=true)
        Z_dep_sparse = sparse(Z_dep)

        # Check selected entries match true inverse
        for (i, j, v) in zip(findnz(Z_dep_sparse)...)
            @test v ≈ A_inv[i, j] atol=1e-10
        end
    end

    @testset "Single-threaded fallback" begin
        # Even with parallel=true, should work correctly when nthreads==1
        # This is tested implicitly by the algorithm falling back to sequential
        n = 500
        A = make_supernodal_spd(n, 0.005)
        F = cholesky(A)

        # This should work regardless of thread count
        Z, p = selinv(F; parallel=true)
        @test size(Z) == size(A)
    end

    @testset "selinv_diag with parallel" begin
        n = 500
        A = make_supernodal_spd(n, 0.005)

        # Compare parallel and sequential
        d_seq = selinv_diag(A; parallel=false)
        d_par = selinv_diag(A; parallel=true)

        @test d_seq ≈ d_par

        # Verify against true diagonal
        A_inv = inv(Matrix(A))
        @test d_par ≈ diag(A_inv)
    end
end
