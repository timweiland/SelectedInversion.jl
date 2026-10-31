```@meta
CurrentModule = SelectedInversion
```

# SelectedInversion

Quickly compute selected entries of the inverse of a sparse matrix.

## Introduction

Sparse matrices are one of the pillars of scientific computing. Sparse factorization methods allow us to solve linear systems involving the inverse efficiently. But in some applications, we might need a bunch of entries of the inverse. Selected inversion algorithms efficiently compute those entries of the inverse that correspond to non-zero entries in the factorization.

SelectedInversion.jl directly interfaces with CHOLMOD-based Cholesky factorizations, which are the default for sparse symmetric positive-definite matrices in Julia, and with sparse LU factorizations (UMFPACK, or KLU through a package extension) for unsymmetric, indefinite and complex matrices.

## Installation

SelectedInversion.jl is a registered Julia package.
To install it:

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Launch the Julia REPL and type `] add SelectedInversion`.

## SelInv API

Make sure to also check the [Tutorial](@ref).

```@docs
selinv
selinv_diag
```

## Sparse LU (unsymmetric, indefinite and complex matrices)

For a square nonsingular sparse matrix that is not symmetric positive definite,
factor it with UMFPACK (`lu(A)`) or KLU (`klu(A)` from KLU.jl, through a package
extension) and request the entries of the inverse you need as the stored pattern
of a sparse matrix `B`. Results are indexed like `A`: the row scaling, the
independent row and column permutations and KLU's block triangular form are
undone internally. The algorithm is the scalar form of Algorithm 1 in
[Jacquelin, Lin and Yang (2018)](https://arxiv.org/abs/1708.04539); the entries
computed are the closure of the requested ones under the recurrence's
dependencies on the stored factors. The one-shot functions are
`selinv(F, B)` and `selinv_diag(F)` (documented above); for repeated
refactorizations of matrices with the same pattern, reuse a workspace:

```@docs
LUSelInvWorkspace
selinv!
```
