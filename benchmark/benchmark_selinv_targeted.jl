using BenchmarkTools
using SelectedInversion
using MatrixMarket
using LinearAlgebra, SparseArrays
using SuiteSparseMatrixCollection

# Two representative matrices:
#   crystm03  - medium, dense supernodes (our optimization helps)
#   ecology2  - large, very sparse supernodes (regression case)
TARGET_MATS = ["crystm03", "shipsec1", "pwtk", "ecology2"]

ssmc = ssmc_db()
mats = ssmc[
    (ssmc.numerical_symmetry .== 1) .&
    (ssmc.positive_definite .== true) .&
    (ssmc.real .== true) .&
    (ssmc.name .∈ Ref(TARGET_MATS)),
    :,
]
paths = fetch_ssmc(mats, format = "MM")

for (i, path) in enumerate(paths)
    name = mats.name[i]
    mat_path = joinpath(path, "$(name).mtx")
    A = MatrixMarket.mmread(mat_path)
    println("$name (N=$(size(A,1)), nnz=$(nnz(A))):")

    C = cholesky(A)
    selinv(C) # warmup

    t = @belapsed selinv($C)
    println("  selinv: $(round(t * 1000; digits=2)) ms")
end
