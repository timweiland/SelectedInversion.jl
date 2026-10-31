using SelectedInversion
using KLU
using LinearAlgebra, SparseArrays
using Random
using Test

# Largest deviation of the stored entries of `Z` from the dense reference `E`,
# relative to the largest entry of `E`.
function klu_relerr(Z::SparseMatrixCSC, E::AbstractMatrix)
    i, j, v = findnz(Z)
    err = maximum(t -> abs(v[t] - E[i[t], j[t]]), eachindex(v); init = 0.0)
    return err / max(1.0, maximum(abs, E))
end

klu_pattern(A) = SparseMatrixCSC(size(A)..., copy(A.colptr), copy(rowvals(A)), ones(Bool, nnz(A)))
klu_full(n) = sparse(ones(Bool, n, n))

function klu_augmented(D, B, W)
    r = size(B, 1)
    E = spdiagm(0 => ones(r))
    C = B / sqrt(2)
    return [D C' C'; C E + W / 2 -W / 2; C -W / 2 -E + W / 2]
end

# A nonsingular matrix whose block triangular form has several blocks coupled by
# a nonzero off-diagonal part: random diagonal blocks, a strictly block-upper
# coupling, then independent random row and column permutations.
function reducible(rng, sizes; T = Float64)
    n = sum(sizes)
    A = spzeros(T, n, n)
    offsets = cumsum([0; sizes])
    for (b, m) in enumerate(sizes)
        r = (offsets[b] + 1):offsets[b + 1]
        A[r, r] = sprandn(rng, T, m, m, min(1.0, 2 / m)) + spdiagm(0 => 2 .+ rand(rng, m))
        for c in (b + 1):length(sizes)
            s = (offsets[c] + 1):offsets[c + 1]
            A[r, s] = sprandn(rng, T, m, sizes[c], 0.3)
        end
    end
    return A[randperm(rng, n), randperm(rng, n)]
end

@testset "Pivoted sparse LU (KLU)" begin
    @testset "Augmented scalar systems with zero or indefinite pivots" begin
        n_cases = 0
        for d in (-1.0, 0.0, 1.0, 2.0), b in (0.1, 1.0, 2.0), w in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
            d + b^2 * w > 0 || continue
            n_cases += 1
            K = klu_augmented(sparse([d;;]), sparse([b;;]), sparse([w;;]))
            F = klu(K)
            E = inv(Matrix(K))
            @test klu_relerr(selinv(F, K), E) < 1.0e-12
            @test klu_relerr(selinv(F, klu_full(3)), E) < 1.0e-12
            @test selinv_diag(F) ≈ diag(E)
        end
        @test n_cases == 43
    end

    @testset "Block triangular form with off-diagonal blocks" begin
        # Symmetric pattern, zero diagonal entry: two BTF blocks coupled by F.
        A = sparse([0.0 1; 1 1])
        F = klu(A)
        @test F.nblocks == 2
        @test nnz(F.F) > 0
        @test klu_relerr(selinv(F, klu_full(2)), inv(Matrix(A))) < 1.0e-14
        @test selinv_diag(F) ≈ diag(inv(Matrix(A)))

        rng = Xoshiro(17)
        seen_coupled = 0
        for sizes in ([3, 4], [1, 5, 2, 6], [8, 1, 1, 9, 4], [20, 15, 25])
            A = reducible(rng, sizes)
            F = klu(A)
            seen_coupled += F.nblocks > 1 && nnz(F.F) > 0
            E = inv(Matrix(A))
            n = size(A, 1)
            # Requests inside and outside the transposed pattern of A reach the
            # strictly block-upper part of the inverse, which needs F.
            @test klu_relerr(selinv(F, A), E) < 1.0e-9
            @test klu_relerr(selinv(F, sparse(A')), E) < 1.0e-9
            @test klu_relerr(selinv(F, klu_full(n)), E) < 1.0e-9
            @test selinv_diag(F) ≈ diag(E)
        end
        @test seen_coupled >= 3
    end

    @testset "Random unsymmetric matrices and row scaling" begin
        rng = Xoshiro(29)
        seen_distinct_perms = false
        for n in (1, 2, 5, 20, 60), trial in 1:5
            A = sprandn(rng, n, n, min(1.0, 3 / n)) + spdiagm(0 => randn(rng, n))
            # Widely different row scales, so the scale factors matter: applying
            # them in the wrong order gives errors of the size of their ratios.
            # The scaled matrices are ill-conditioned (up to about 1e14), so the
            # tolerance is the forward-error scale eps * cond.
            A = spdiagm(0 => exp10.(rand(rng, -6:6, n))) * A
            F = klu(A)
            seen_distinct_perms |= F.p != F.q
            E = inv(Matrix(A))
            tol = max(1.0e-10, eps() * cond(Matrix(A)))
            @test klu_relerr(selinv(F, A), E) < tol
            @test klu_relerr(selinv(F, sparse(A')), E) < tol
            @test klu_relerr(selinv(F, sprand(rng, Bool, n, n, 0.2)), E) < tol
            @test norm(selinv_diag(F) - diag(E), Inf) < tol * max(1.0, maximum(abs, E))
        end
        @test seen_distinct_perms
    end

    @testset "Complex matrices and 32-bit indices" begin
        rng = Xoshiro(31)
        for n in (3, 20, 50)
            A = sprandn(rng, ComplexF64, n, n, 3 / n) + spdiagm(0 => randn(rng, ComplexF64, n))
            F = klu(A)
            Z = selinv(F, A)
            @test eltype(Z) == ComplexF64
            @test klu_relerr(Z, inv(Matrix(A))) < 1.0e-9
        end
        A = reducible(rng, [4, 6, 3]; T = ComplexF64)
        @test klu_relerr(selinv(klu(A), klu_full(13)), inv(Matrix(A))) < 1.0e-9
        A = sprandn(rng, 30, 30, 0.1) + spdiagm(0 => randn(rng, 30))
        A32 = SparseMatrixCSC{Float64, Int32}(A)
        F = klu(A32)
        @test F isa KLU.KLUFactorization{Float64, Int32}
        @test klu_relerr(selinv(F, A32), inv(Matrix(A))) < 1.0e-9
    end

    @testset "Workspace reuse across klu!" begin
        rng = Xoshiro(37)
        A1 = reducible(rng, [10, 15, 12])
        F = klu(A1)
        ws = LUSelInvWorkspace(F, A1)
        dest = similar(A1)
        selinv!(dest, ws, F)
        @test klu_relerr(dest, inv(Matrix(A1))) < 1.0e-9
        A2 = copy(A1)
        nonzeros(A2) .*= 1 .+ 0.01 .* rand(rng, nnz(A2))
        klu!(F, A2)
        selinv!(dest, ws, F)
        @test klu_relerr(dest, inv(Matrix(A2))) < 1.0e-9
        @test ws.preparations == 1
        # A factorization from another solver re-prepares the same workspace.
        selinv!(dest, ws, lu(A2))
        @test ws.preparations == 2
        @test klu_relerr(dest, inv(Matrix(A2))) < 1.0e-9
        @test_throws ArgumentError selinv!(similar(A1)[:, end:-1:1], ws, F)
        @test_throws ArgumentError selinv_diag(F; depermute = false)
    end

    @testset "Bounded fill: computed entries grow linearly" begin
        function arrow(n)
            D = spdiagm(0 => [0.0; ones(n - 1)])
            B = sparse(fill(1, n), 1:n, [1.0; fill(0.01, n - 1)], 1, n)
            return klu_augmented(D, B, sparse([1.0;;]))
        end
        ratios = Float64[]
        for n in (512, 4096, 32768)
            K = arrow(n)
            B = klu_pattern(K) .| sparse(I, size(K)...)
            F = klu(K)
            ws = LUSelInvWorkspace(F, B)
            push!(ratios, (length(ws.lo_rowval) + length(ws.up_colval) + ws.n) / size(K, 1))
            Z = float(copy(B))
            selinv!(Z, ws, F)
            @test Z[1, 1] ≈ 1 + (n - 1) * 1.0e-4
            @test Z[2, 2] ≈ 1
            @test sum(nonzeros(K .* copy(transpose(Z)))) ≈ size(K, 1)
        end
        @test maximum(ratios) <= 1.1 * minimum(ratios)
    end
end
