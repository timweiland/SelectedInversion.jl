using SelectedInversion
using SparseArrays, LinearAlgebra
using BenchmarkTools

# Generate SPD sparse matrix that produces supernodal factorization
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

N = 2000
println("Generating SPD matrix of size $N...")
A = make_supernodal_spd(N, 15)
println("  nnz(A) = $(nnz(A))")

println("Computing Cholesky and selected inverse...")
F = cholesky(A)
Zp = selinv(F)
Z = Zp.Z
p = Zp.p

# Permute A to match Z's layout
B = A[p, p]
println("  nnz(B) = $(nnz(B))")

# Convert to sparse for reference
Z_sparse = sparse(Z)
println("  nnz(Z_sparse) = $(nnz(Z_sparse))")

# Correctness check
ref = dot(Z_sparse, B)
println("\nReference dot(sparse(Z), B) = $ref")

# Benchmark: dot with SupernodalMatrix (current fallback or specialized)
println("\n--- dot(Z::SupernodalMatrix, B::SparseMatrixCSC) ---")
result_supernodal = dot(Z, B)
println("Result: $result_supernodal")
println("Match: $(isapprox(result_supernodal, ref; rtol=1e-10))")
@btime dot($Z, $B)

# Benchmark: sparse conversion + dot (the naive alternative)
println("\n--- sparse(Z) + dot(sparse(Z), B) ---")
@btime dot(sparse($Z), $B)

# Benchmark: dot with pre-computed sparse (best-case reference)
println("\n--- dot(Z_sparse::SparseMatrixCSC, B::SparseMatrixCSC) [pre-converted] ---")
@btime dot($Z_sparse, $B)
