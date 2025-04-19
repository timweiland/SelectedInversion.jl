using MatrixMarket, SuiteSparseMatrixCollection

mat_names = ["494_bus", "parabolic_fem"]
ssmc = ssmc_db()
mats_df = ssmc[ssmc.name .∈ Ref(mat_names), :]
paths = fetch_ssmc(mats_df, format = "MM")
paths = [joinpath(path, "$(mats_df.name[i]).mtx") for (i, path) in enumerate(paths)]
A = MatrixMarket.mmread(paths[1]) # 494_bus
B = MatrixMarket.mmread(paths[2]) # parabolic_fem
size(A), size(B)

A

using SelectedInversion
Z, p = selinv(A)
Z

A_inv = inv(Array(A))

A_inv[42, 172], Z[42, 172]

A_inv[42, 172], Z[invperm(p), invperm(p)][42, 172]

Z, _ = selinv(A; depermute=true)
Z

A_inv[42, 172], Z[42, 172]

B

using LinearAlgebra
C = cholesky(B)

Z, p = selinv(C; depermute=true)
Z

e5 = zeros(size(B, 2))
e5[5] = 1.
(B \ e5)[end], Z[end, 5]

diag(Z)

using SparseArrays
sparse(Z)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
