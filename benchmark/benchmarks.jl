using BenchmarkTools
using SelectedInversion

using MatrixMarket
using LinearAlgebra, SparseArrays

include("matrix_suppliers.jl")

const SUITE = BenchmarkGroup()
SUITE["SelInv_spd_supernodal"] = BenchmarkGroup(["sequential", "supernodal", "spd"])
SUITE["Cholesky_supernodal"] = BenchmarkGroup(["cholesky", "supernodal"])

names, spd_mat_paths = get_suitesparse_spd()
n_total = length(names)
for (idx, (name, mat_path)) in enumerate(zip(names, spd_mat_paths))
    A = MatrixMarket.mmread(mat_path)
    println("[$idx/$n_total] $name (N=$(size(A,1)), nnz=$(nnz(A)))...")
    C = cholesky(A)
    print("  Cholesky... ")
    SUITE["Cholesky_supernodal"][name] = @benchmark cholesky($A)
    println("done")
    print("  SelInv...   ")
    SUITE["SelInv_spd_supernodal"][name] = @benchmark selinv($C)
    println("done")
end
