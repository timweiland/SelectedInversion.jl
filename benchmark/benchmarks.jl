using BenchmarkTools
using SelectedInversion

using MatrixMarket
using LinearAlgebra, SparseArrays

include("matrix_suppliers.jl")

const SUITE = BenchmarkGroup()
SUITE["SelInv_spd_supernodal"] = BenchmarkGroup(["sequential", "supernodal", "spd"])
SUITE["Cholesky_supernodal"] = BenchmarkGroup(["cholesky", "supernodal"])

names, spd_mat_paths = get_suitesparse_spd()
for (name, mat_path) in zip(names, spd_mat_paths)
    A = MatrixMarket.mmread(mat_path)
    C = cholesky(A)
    SUITE["Cholesky_supernodal"][name] = @benchmark cholesky($A)
    SUITE["SelInv_spd_supernodal"][name] = @benchmark selinv($C)
end
