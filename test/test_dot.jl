using SelectedInversion
using LinearAlgebra, SparseArrays
using Test

function make_supernodal_spd(n, entries_per_row)
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

@testset "dot(SupernodalMatrix, SparseMatrixCSC)" begin
    for N in [50, 200, 500]
        A = make_supernodal_spd(N, 15)
        F = cholesky(A)

        is_super = Bool(unsafe_load(pointer(F)).is_super)
        if !is_super
            continue
        end

        @testset "N=$N" begin
            # Non-depermuted selected inverse
            Zp = selinv(F; depermute = false)
            Z = Zp.Z
            p = Zp.p
            Z_sparse = sparse(Z)
            B = A[p, p]

            @testset "dot(Z, B) matches sparse reference" begin
                ref = dot(Z_sparse, B)
                @test dot(Z, B) ≈ ref
            end

            @testset "dot(B, Z) matches sparse reference" begin
                ref = dot(Z_sparse, B)
                @test dot(B, Z) ≈ ref
            end

            @testset "dot(Z, Z_sparse) matches sparse reference" begin
                ref = dot(Z_sparse, Z_sparse)
                @test dot(Z, Z_sparse) ≈ ref
            end

            @testset "dot with identity-like sparse matrix" begin
                I_sp = sparse(1.0I, N, N)
                ref = dot(Z_sparse, I_sp)
                @test dot(Z, I_sp) ≈ ref
            end

            @testset "dot with zero matrix" begin
                O = spzeros(N, N)
                @test dot(Z, O) == 0.0
            end

            # Depermuted selected inverse
            Zp_dep = selinv(F; depermute = true)
            Z_dep = Zp_dep.Z
            Z_dep_sparse = sparse(Z_dep)

            @testset "depermuted dot(Z, A) matches sparse reference" begin
                ref = dot(Z_dep_sparse, A)
                @test dot(Z_dep, A) ≈ ref
            end

            @testset "depermuted dot(Z, B) matches sparse reference" begin
                ref = dot(Z_dep_sparse, B)
                @test dot(Z_dep, B) ≈ ref
            end
        end
    end
end
