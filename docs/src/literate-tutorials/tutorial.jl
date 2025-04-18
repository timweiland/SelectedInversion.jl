# # Tutorial
#
# ## Introduction
# In this tutorial, we're going to explore how to use SelectedInversion.jl to
# compute the selected inverse of two different sparse symmetric positive definite
# matrices.
#
# ## Problem setup
# We're going to grab two matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu).
# Refer to the website for more information on these matrices and where they came from.
#
# Feel free to skip ahead, as this part is not related directly to SelectedInversion.jl.
using MatrixMarket, SuiteSparseMatrixCollection

mat_names = ["494_bus", "parabolic_fem"]
ssmc = ssmc_db()
mats_df = ssmc[ssmc.name .∈ Ref(mat_names), :]
paths = fetch_ssmc(mats_df, format = "MM")
paths = [joinpath(path, "$(mats_df.name[i]).mtx") for (i, path) in enumerate(paths)]
A = MatrixMarket.mmread(paths[1]) # 494_bus
B = MatrixMarket.mmread(paths[2]) # parabolic_fem
size(A), size(B)

# ## SelInv
# We're going to start by tackling the smaller of the two matrices.
A

# Let's compute the selected inverse of `A`.
using SelectedInversion
Z, p = selinv(A)
Z

# Now the nonzero entries of `Z` correspond to entries in `inv(A)`.
#
# `p` is a permutation vector.
# Sparse Cholesky factorizations reorder the rows and columns of a matrix to
# reduce fill-in in the Cholesky factor.
# `selinv` computes its entries according to this permutation.
#
# Concretely, this means that if we want to get a specific entry of the inverse,
# we need to apply the correct permutation first.
#
# To test this, we're going to compute the dense full inverse.
# This is still feasible for such a small matrix, but not recommended in general.
A_inv = inv(Array(A))

# Compare the values of `Z` and `A_inv` at some arbitrary index.
# They're not going to match:
A_inv[42, 172], Z[42, 172]

# But if we permute Z first, they do match:
A_inv[42, 172], Z[invperm(p), invperm(p)][42, 172]

# If your use case calls for this kind of depermuted access, you can make life
# easier with the `depermute` keyword:
Z, _ = selinv(A; depermute=true)
Z

# Now the nonzero entries of `Z` directly give you the corresponding entries of
# `inv(A)`.
A_inv[42, 172], Z[42, 172]

# ## Supernodal setting
# Now, let's tackle the bigger of the two matrices.
B

# We could directly apply `selinv` to `B`.
# But if we have access to a Cholesky factorization of `B`, we can pass that
# to `selinv` instead, which is going to be faster because `selinv` would have
# computed a Cholesky internally anyways.
#
# So just to prove a point, let's first compute a Cholesky factorization:
using LinearAlgebra
C = cholesky(B)

# As we can see, this is a *supernodal* Cholesky factorization.
# Supernodal factorizations chunk contiguous columns with an identical sparsity
# pattern. Computations may then leverage BLAS for these chunks, which can speed
# things up quite a lot.
#
# SelInv also uses the supernodal structure internally.
# As a result, the return type of `Z` is now different.
# Let's compute the selected inverse.
Z, p = selinv(C; depermute=true)
Z

# `Z` is now a `SupernodalMatrix`, which is a custom type defined in
# SelectedInversion.jl.
#
# It's a subtype of `AbstractMatrix`, so you can index into it as you would expect.
# Let's check the value of some arbitrary entry.
e5 = zeros(size(B, 2))
e5[5] = 1.
(B \ e5)[end], Z[end, 5]

# The diagonal might be particularly relevant to some applications:
diag(Z)

# It's also possible to convert `Z` into a sparse matrix.
# But this is fairly slow and eats up a lot of memory.
# As such, it should be avoided unless it's truly necessary.
using SparseArrays
sparse(Z)

# ## Conclusion
# SelectedInversion.jl lets you compute the selected inverse of a sparse symmetric
# positive definite matrix efficiently.
# Where applicable, it makes use of supernodal factorizations and thus scales to
# matrices with more than a million columns.
#
# As of now, it does not support unsymmetric matrices and does not explicitly
# make use of parallelization along the elimination tree.
# If you're interested in helping develop these features, feel free to open an
# issue or a pull request on GitHub.
