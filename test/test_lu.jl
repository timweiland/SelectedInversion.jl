using SelectedInversion
using LinearAlgebra, SparseArrays
using Random
using Test

# Largest deviation of `Z` (a sparse result) from the dense reference `E` on
# Z's stored entries, relative to the largest entry of `E`.
function lu_relerr(Z::SparseMatrixCSC, E::AbstractMatrix)
    i, j, v = findnz(Z)
    err = maximum(t -> abs(v[t] - E[i[t], j[t]]), eachindex(v); init = 0.0)
    return err / max(1.0, maximum(abs, E))
end

lu_dense_check(A, B) = lu_relerr(selinv(lu(A), B), inv(Matrix(A)))

pattern_of(A) = SparseMatrixCSC(size(A)..., copy(A.colptr), copy(rowvals(A)), ones(Bool, nnz(A)))

# Rotated augmented system whose Schur complement is D + B' W B; its local
# pivots can be zero or indefinite even when the Schur complement is positive
# definite.
function augmented(D, B, W)
    r = size(B, 1)
    E = spdiagm(0 => ones(r))
    C = B / sqrt(2)
    return [D C' C'; C E + W / 2 -W / 2; C -W / 2 -E + W / 2]
end

# Number of computed entries of X = inv((R A)[P, Q]) that are not in the stored
# pattern of transpose(L + U) or on the diagonal: entries the closure added
# because UMFPACK dropped exactly cancelled factor entries, or because the
# request lies outside transpose(L + U).
function closure_excess(ws::LUSelInvWorkspace, F)
    LU = F.L + F.U
    count = 0
    for i in 1:size(F, 1), s in ws.lo_colptr[i]:(ws.lo_colptr[i + 1] - 1)
        count += iszero(LU[i, ws.lo_rowval[s]])
    end
    for r in 1:size(F, 1), s in ws.up_rowptr[r]:(ws.up_rowptr[r + 1] - 1)
        count += iszero(LU[ws.up_colval[s], r])
    end
    return count
end

# tr(K * Z) from Z stored at the pattern of transpose(K).
trace_product(K, Z) = sum(nonzeros(K .* copy(transpose(Z))))

closure_size(ws::LUSelInvWorkspace) = length(ws.lo_rowval) + length(ws.up_colval) + ws.n

# Effects coupled through cells: cell c touches effects c, c+1, …, c+width-1
# (cyclically when `cyclic`), so neighbouring cells overlap.
function overlapping_cells(rng, m, width; cyclic = false)
    I, J, V = Int[], Int[], Float64[]
    r = cyclic ? m : m - width + 1
    for c in 1:r, t in 0:(width - 1)
        push!(I, c)
        push!(J, cyclic ? mod1(c + t, m) : c + t)
        push!(V, randn(rng))
    end
    return sparse(I, J, V, r, m)
end

function overlapping_augmented(rng, m, width; cyclic = false)
    B = overlapping_cells(rng, m, width; cyclic)
    r = size(B, 1)
    # Zero, positive and negative prior curvature; positive and negative cell
    # curvature.
    D = spdiagm(0 => rand(rng, (0.0, 0.0, 1.0, -0.5), m))
    W = spdiagm(0 => rand(rng, (2.0, 1.0, 0.5, -0.25), r))
    return augmented(D, B, W)
end

@testset "Pivoted sparse LU (UMFPACK)" begin
    @testset "Augmented scalar systems with zero or indefinite pivots" begin
        n_cases = 0
        n_excess = 0
        for d in (-1.0, 0.0, 1.0, 2.0), b in (0.1, 1.0, 2.0), w in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
            d + b^2 * w > 0 || continue
            n_cases += 1
            K = augmented(sparse([d;;]), sparse([b;;]), sparse([w;;]))
            F = lu(K)
            E = inv(Matrix(K))
            @test lu_relerr(selinv(F, K), E) < 1.0e-12
            full = sparse(ones(Bool, 3, 3))
            ws = LUSelInvWorkspace(F, full)
            @test lu_relerr(selinv!(float(copy(full)), ws, F), E) < 1.0e-12
            n_excess += closure_excess(LUSelInvWorkspace(F, K), F) > 0
        end
        @test n_cases == 43
        # Some of these factorizations drop an exactly cancelled entry that the
        # recurrence needs; the closure must have restored it.
        @test n_excess > 0
    end

    @testset "Explicit zeros and zero diagonal entries are requested" begin
        K = augmented(sparse([0.0;;]), sparse([1.0;;]), sparse([1.0;;]))
        @test K[1, 1] == 0
        B = copy(K)
        B[1, 1] = 1.0
        B.nzval .= 0                   # every stored entry is an explicit zero
        Z = selinv(lu(K), B)
        @test Z.colptr == B.colptr && Z.rowval == B.rowval
        @test lu_relerr(Z, inv(Matrix(K))) < 1.0e-12
        @test Z[1, 1] ≈ inv(Matrix(K))[1, 1]
    end

    @testset "Random unsymmetric matrices against dense inverses" begin
        rng = Xoshiro(20261031)
        seen_distinct_perms = false
        seen_scaling = false
        for n in (1, 2, 5, 20, 60), trial in 1:5
            A = sprandn(rng, n, n, min(1.0, 3 / n)) + spdiagm(0 => randn(rng, n))
            F = lu(A)
            seen_distinct_perms |= F.p != F.q
            seen_scaling |= !all(isone, F.Rs)
            E = inv(Matrix(A))
            @test lu_relerr(selinv(F, A), E) < 1.0e-9
            @test lu_relerr(selinv(F, sparse(A')), E) < 1.0e-9
            @test lu_relerr(selinv(F, sprand(rng, Bool, n, n, 0.2)), E) < 1.0e-9
            @test lu_relerr(selinv(F, sparse(ones(Bool, n, n))), E) < 1.0e-9
            @test selinv_diag(F) ≈ diag(E)
        end
        @test seen_distinct_perms
        @test seen_scaling
    end

    @testset "Zero diagonal requiring off-diagonal pivots" begin
        rng = Xoshiro(7)
        for n in (4, 10, 40)
            # Permuted nonsingular matrix: the diagonal is structurally zero.
            perm = circshift(1:n, 1)
            A = sparse(1:n, perm, 1.0 .+ rand(rng, n), n, n) +
                sprandn(rng, n, n, 1 / n) .* 0.1
            A = dropzeros(A - spdiagm(0 => diag(A)))
            F = lu(A)
            @test F.p != F.q
            @test lu_dense_check(A, pattern_of(A) + sparse(I, n, n)) < 1.0e-10
        end
    end

    @testset "Complex matrices" begin
        rng = Xoshiro(11)
        for n in (3, 20, 50)
            A = sprandn(rng, ComplexF64, n, n, 3 / n) + spdiagm(0 => randn(rng, ComplexF64, n))
            F = lu(A)
            E = inv(Matrix(A))
            Z = selinv(F, A)
            @test eltype(Z) == ComplexF64
            @test lu_relerr(Z, E) < 1.0e-9
            @test selinv_diag(F) ≈ diag(E)
            # Inverse, not adjoint: no conjugation anywhere.
            Zt = selinv(lu(sparse(transpose(A))), sparse(transpose(A)))
            @test lu_relerr(Zt, transpose(E)) < 1.0e-9
        end
        K = augmented(sparse([0.0im;;]), sparse([1.0 + 1im;;]), sparse([1.0 - 2im;;]))
        @test lu_dense_check(K, K) < 1.0e-12
    end

    @testset "32-bit indices" begin
        rng = Xoshiro(3)
        A = sprandn(rng, 30, 30, 0.1) + spdiagm(0 => randn(rng, 30))
        A32 = SparseMatrixCSC{Float64, Int32}(A)
        F = lu(A32)
        @test F isa SparseArrays.UMFPACK.UmfpackLU{Float64, Int32}
        @test lu_relerr(selinv(F, A32), inv(Matrix(A))) < 1.0e-9
        Ac = SparseMatrixCSC{ComplexF64, Int32}(A + im * spdiagm(0 => ones(30)))
        @test lu_relerr(selinv(lu(Ac), Ac), inv(Matrix(Ac))) < 1.0e-9
    end

    @testset "Row scaling by division (tiny row sums)" begin
        # UMFPACK divides by the scale factors instead of multiplying when a row
        # sum is tiny. SparseArrays' F.Rs ignored this before
        # JuliaSparse/SparseArrays.jl#899, so the factors are read directly.
        A = sparse([1.0e-14 2.0e-14 0; 1 3 1; 0 1 4.0])
        F = lu(A)
        factors = SelectedInversion._read_factors!(
            SelectedInversion._LUFactors{Float64, Int}(), F,
        )
        @test iszero(factors.do_recip[])
        r = SelectedInversion._row_multipliers!(Float64[], factors)
        @test Matrix(F.L * F.U) ≈ (r .* Matrix(A))[F.p, F.q]
        @test lu_dense_check(A, sparse(ones(Bool, 3, 3))) < 1.0e-12
        @test selinv_diag(F) ≈ diag(inv(Matrix(A)))
    end

    @testset "Overlapping cells against dense inverses" begin
        rng = Xoshiro(5)
        for (m, width, cyclic) in ((12, 2, false), (30, 3, true), (60, 4, false))
            K = overlapping_augmented(rng, m, width; cyclic)
            F = lu(K)
            E = inv(Matrix(K))
            @test lu_relerr(selinv(F, K), E) < 1.0e-9
            @test selinv_diag(F) ≈ diag(E)
        end
    end

    @testset "Workspace reuse and pivot changes" begin
        rng = Xoshiro(9)
        n = 40
        A1 = sprandn(rng, n, n, 0.1) + spdiagm(0 => 4 .+ rand(rng, n))
        F = lu(A1)
        ws = LUSelInvWorkspace(F, A1)
        @test ws.preparations == 1
        dest = similar(A1)
        selinv!(dest, ws, F)
        @test lu_relerr(dest, inv(Matrix(A1))) < 1.0e-10
        @test ws.preparations == 1

        # Same pattern, new values, same pivots: numeric phase only. The only
        # allocation is the O(n) temporary workspace UMFPACK's get_numeric
        # takes from Julia's allocator; it does not grow with the fill.
        A2 = copy(A1)
        nonzeros(A2) .*= 1 .+ 0.01 .* rand(rng, nnz(A2))
        lu!(F, A2)
        selinv!(dest, ws, F)
        @test lu_relerr(dest, inv(Matrix(A2))) < 1.0e-10
        @test ws.preparations == 1
        @test (@allocated selinv!(dest, ws, F)) <= 32 * n + 1024

        # Values that force different row pivots: the workspace is prepared
        # again and the result stays correct.
        p_before = F.p
        A3 = copy(A2)
        for j in 1:n
            A3[j, j] = 1.0e-8
        end
        lu!(F, A3)
        @test F.p != p_before
        selinv!(dest, ws, F)
        @test ws.preparations == 2
        @test lu_relerr(dest, inv(Matrix(A3))) < 1.0e-8

        # Independent workspaces on one factorization.
        ws2 = LUSelInvWorkspace(F, A1)
        dest2 = similar(A1)
        selinv!(dest2, ws2, F)
        @test nonzeros(dest2) == nonzeros(dest)
    end

    @testset "Argument checks" begin
        A = sparse([2.0 1; 1 3])
        F = lu(A)
        ws = LUSelInvWorkspace(F, A)
        @test_throws ArgumentError selinv!(sparse([1.0 0; 0 1]), ws, F)
        @test_throws DimensionMismatch selinv!(similar(A), ws, lu(sparse(4.0I, 3, 3)))
        @test_throws DimensionMismatch selinv(F, sparse(1.0I, 3, 3))
        @test_throws ArgumentError selinv_diag(F; depermute = false)
        S = lu(sparse([1.0 1; 1 1]); check = false)
        @test_throws SingularException selinv(S, sparse([1.0 1; 1 1]))
        R = sparse([1.0 0 0; 0 1 0; 0 0 1; 1 1 1])
        @test_throws DimensionMismatch selinv(lu(R; check = false), R)
    end

    @testset "Bounded fill: computed entries grow linearly" begin
        # Arrow system with a zero first prior curvature (as in the consumer's
        # gate) and overlapping-cell chains. The computed set must grow linearly
        # with the problem size.
        function arrow(n)
            D = spdiagm(0 => [0.0; ones(n - 1)])
            B = sparse(fill(1, n), 1:n, [1.0; fill(0.01, n - 1)], 1, n)
            return augmented(D, B, sparse([1.0;;]))
        end
        ratios = Float64[]
        for n in (512, 4096, 32768)
            K = arrow(n)
            @test iszero(K[1, 1])            # not stored, but requested below
            B = pattern_of(K) .| sparse(I, size(K)...)
            F = lu(K)
            ws = LUSelInvWorkspace(F, B)
            push!(ratios, closure_size(ws) / size(K, 1))
            Z = float(copy(B))
            selinv!(Z, ws, F)
            @test Z[1, 1] ≈ 1 + (n - 1) * 1.0e-4
            @test Z[2, 2] ≈ 1
            @test trace_product(K, Z) ≈ size(K, 1)   # tr(K * inv(K))
        end
        @test maximum(ratios) <= 1.1 * minimum(ratios)

        rng = Xoshiro(13)
        ratios = Float64[]
        for m in (1000, 8000, 32000)
            K = overlapping_augmented(rng, m, 3)
            F = lu(K)
            ws = LUSelInvWorkspace(F, K)
            push!(ratios, closure_size(ws) / size(K, 1))
            Z = similar(K)
            selinv!(Z, ws, F)
            @test trace_product(K, Z) ≈ size(K, 1) rtol = 1.0e-8
            # Spot-check whole columns against solves.
            for j in (1, m ÷ 2, size(K, 1))
                e = zeros(size(K, 1))
                e[j] = 1
                x = F \ e
                for t in nzrange(Z, j)
                    @test Z.nzval[t] ≈ x[rowvals(Z)[t]] atol = 1.0e-10 * norm(x, Inf)
                end
            end
        end
        @test maximum(ratios) <= 1.5 * minimum(ratios)
    end
end
