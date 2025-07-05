```@meta
CurrentModule = SelectedInversion
```

# SelectedInversion

Quickly compute selected entries of the inverse of a sparse matrix.

## Introduction

Sparse matrices are one of the pillars of scientific computing. Sparse factorization methods allow us to solve linear systems involving the inverse efficiently. But in some applications, we might need a bunch of entries of the inverse. Selected inversion algorithms efficiently compute those entries of the inverse that correspond to non-zero entries in the factorization.

SelectedInversion.jl directly interfaces with CHOLMOD-based Cholesky factorizations, which are the default for sparse symmetric positive-definite matrices in Julia.

## Installation

SelectedInversion.jl is not yet a registered Julia package.
Until it is, you can install it from this GitHub repository.
To do so:

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Launch the Julia REPL and type `] add https://github.com/timweiland/SelectedInversion.jl`. 

## SelInv API

Make sure to also check the [Tutorial](@ref).

```@docs
selinv
selinv_diag
```
