using SelectedInversion
using LinearAlgebra, SparseArrays
using Random
using Test

# Deterministic sparse SPD matrix that yields a supernodal factorization.
function extract_supernodal_spd(n, entries_per_row)
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    for i in 1:n
        for k in 0:(entries_per_row - 1)
            j = mod(i + k * 7 + k^2, n) + 1
            push!(I_idx, i)
            push!(J_idx, j)
            push!(V, 1.0 + mod(i * 3 + j * 5, 10) / 10.0)
        end
    end
    A = sparse(I_idx, J_idx, V, n, n)
    return A * A' + 5I
end

# Tridiagonal 1D Laplacian — yields a simplicial factorization.
laplacian_1d(n) = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))

# Independent reference: read Σ at B's pattern via ordinary indexing.
function mask_reference(Σ, B::SparseMatrixCSC)
    dest = SparseMatrixCSC(
        B.m, B.n, copy(B.colptr), copy(rowvals(B)), zeros(Float64, nnz(B)),
    )
    rv = rowvals(B)
    nz = nonzeros(dest)
    for j in 1:B.n
        for t in nzrange(B, j)
            nz[t] = Σ[rv[t], j]
        end
    end
    return dest
end

# Symmetric random pattern of size n×n with the given density.
function random_pattern(seed, n, density)
    rng = MersenneTwister(seed)
    R = sprand(rng, n, n, density)
    B = R + R' + sparse(1.0I, n, n)
    return dropzeros(B)
end

# Observation-style pattern AᵀA (A is m×n).
function obs_pattern(seed, m, n, density)
    rng = MersenneTwister(seed)
    A = sprand(rng, m, n, density)
    return dropzeros(A' * A)
end

# Allocation probe isolated in a function so locals are concretely typed.
measure_fill_alloc(dest, S, plan) = @allocated selinv_extract!(dest, S, plan)

# Shared assertions for one (S/Z, Σ, B) triple.
function check_extract(SorZ, Σ, B; bit_identical = false)
    Z = selinv_extract(SorZ, B)
    ref = mask_reference(Σ, B)

    # Exact pattern of B.
    @test Z.colptr == B.colptr
    @test rowvals(Z) == rowvals(B)
    @test size(Z) == size(B)

    # Values match Σ at every nonzero of B (including off-pattern → 0).
    @test nonzeros(Z) ≈ nonzeros(ref)
    if bit_identical
        @test maximum(abs, nonzeros(Z) .- nonzeros(ref)) == 0.0
    end

    # In-place into a preallocated dest carrying B's pattern.
    dest = SparseMatrixCSC(B.m, B.n, copy(B.colptr), copy(rowvals(B)), fill(NaN, nnz(B)))
    selinv_extract!(dest, SorZ, B)
    @test nonzeros(dest) ≈ nonzeros(ref)

    return Z, ref
end

@testset "selinv_extract" begin
    @testset "Supernodal" begin
        for A in (extract_supernodal_spd(500, 15), extract_supernodal_spd(1200, 15))
            F = cholesky(A)
            @test Bool(unsafe_load(pointer(F)).is_super)
            N = size(A, 1)

            # Patterns: a random symmetric one and an obs-style AᵀA one.
            B_rand = random_pattern(1, N, 4 / N)
            B_obs = obs_pattern(7, N ÷ 2, N, 6 / N)

            @testset "depermute=true" begin
                S = selinv(F; depermute = true).Z
                @test S isa SupernodalMatrix
                Σ = sparse(S)

                for B in (B_rand, B_obs)
                    Z, ref = check_extract(S, Σ, B; bit_identical = true)
                    # The random pattern must exercise off-pattern (zero) entries.
                    B === B_rand && @test count(iszero, nonzeros(ref)) > 0

                    # Plan-based, allocation-free reuse path.
                    plan = selinv_extract_setup(S, B)
                    dest = selinv_extract(S, B)
                    fill!(nonzeros(dest), NaN)
                    selinv_extract!(dest, S, plan)
                    @test nonzeros(dest) ≈ nonzeros(ref)

                    measure_fill_alloc(dest, S, plan)  # warmup
                    @test measure_fill_alloc(dest, S, plan) == 0
                end
            end

            @testset "depermute=false" begin
                Sp = selinv(F; depermute = false)
                S = Sp.Z
                Σ = sparse(S)  # permuted frame
                for B in (B_rand, B_obs)
                    check_extract(S, Σ, B)
                    plan = selinv_extract_setup(S, B)
                    dest = selinv_extract(S, B)
                    fill!(nonzeros(dest), NaN)
                    selinv_extract!(dest, S, plan)
                    @test nonzeros(dest) ≈ nonzeros(mask_reference(Σ, B))
                    measure_fill_alloc(dest, S, plan)
                    @test measure_fill_alloc(dest, S, plan) == 0
                end
            end
        end
    end

    @testset "Simplicial" begin
        A = laplacian_1d(200)
        F = cholesky(A; perm = 1:size(A, 1))
        @test !Bool(unsafe_load(pointer(F)).is_super)
        N = size(A, 1)
        B_rand = random_pattern(2, N, 6 / N)

        @testset "depermute=true (SparseMatrixCSC)" begin
            Z_full = selinv(F; depermute = true).Z
            @test Z_full isa SparseMatrixCSC
            Σ = sparse(Z_full)
            check_extract(Z_full, Σ, B_rand)
            @test count(iszero, nonzeros(mask_reference(Σ, B_rand))) > 0
        end

        @testset "depermute=false (Symmetric)" begin
            Z_sym = selinv(F; depermute = false).Z
            @test Z_sym isa Symmetric
            Σ = sparse(Z_sym)
            check_extract(Z_sym, Σ, B_rand)
        end
    end
end
