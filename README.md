<h1 align="center">
    SelectedInversion.jl
</h1>

<p align="center">
<strong>🔥 Quickly compute selected entries of the inverse of a sparse matrix</strong>
</p>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/SelectedInversion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/SelectedInversion.jl/dev/)
[![Build Status](https://github.com/timweiland/SelectedInversion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/SelectedInversion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/SelectedInversion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/SelectedInversion.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

</div>

Sparse matrices are one of the pillars of scientific computing.
Sparse factorization methods allow us to solve linear systems involving the inverse efficiently.
But in some applications, we might need *a bunch* of entries of the inverse.
Selected inversion algorithms efficiently compute those entries of the inverse that correspond to non-zero entries in the factorization.

SelectedInversion.jl directly interfaces with CHOLMOD-based Cholesky
factorizations, which are the default for sparse symmetric positive-definite
matrices in Julia.
It also supports factorizations obtained from LDLFactorizations.jl through a
package extension.

The algorithms implemented here are directly based on SelInv [1].
The simplicial formulation is equivalent to the Takahashi recursions [2, 3].

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Contributing](#contributing)

## Installation

SelectedInversion.jl is a registered Julia package.
You can install it directly from the Pkg REPL.
To do so:

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Launch the Julia REPL and type `] add SelectedInversion`. 

## Usage

It's as simple as it can be:

``` julia
using SelectedInversion

A = ... # some sparse matrix
Z, p = selinv(A) # selected inverse and corresponding permutation vector
```

The nonzero entries of `Z` match the corresponding entries in `inv(A[p, p])`.

If you've already computed a Cholesky factorization of your matrix anyway,
you can directly pass it to `selinv` to save the factorization step.

```julia
C = cholesky(A)
Z, p = selinv(C)
```

If you don't care about this whole permutation business, use this:

```julia
Z, _ = selinv(A; depermute=true)
```

Now the nonzero entries of `Z` directly give you the corresponding entries
of `inv(A)`.

### Computing Only the Diagonal

For many applications, you only need the diagonal entries of the inverse matrix.
For example, in Gaussian Markov Random Fields, the diagonal of the inverse precision matrix gives the marginal variances.

```julia
d = selinv_diag(A)  # diagonal of inv(A)
```

This is much more efficient than computing the full selected inverse and then extracting the diagonal, especially for simplicial factorizations.

You can also pass pre-computed factorizations:

```julia
C = cholesky(A)
d = selinv_diag(C)  # same result, but reuses factorization
```

The `selinv_diag` function supports the same `depermute` keyword as `selinv`:

```julia
d_permuted = selinv_diag(A; depermute=false)  # diagonal of permuted inverse
```

## Performance

**tl;dr**: It's pretty fast.

Below is the performance of SelectedInversion.jl on some example problems from
the [SuiteSparse matrix collection](http://sparse.tamu.edu).
The benchmark was run on my laptop (M1 Max with 32 GB RAM).

To put these numbers into perspective, feel free to compare to Table III of [[1]](https://dl.acm.org/doi/abs/10.1145/1916461.1916464).
Note however the hardware difference (they used a Franklin Cray XT4 supercomputer),
as well as the different factorization algorithm (we use CHOLMOD).

| **Problem**<br>`String` | **N**<br>`Int64` | **NNZ**<br>`Int64` | **Factorization time (sec)**<br>`Measurement{Float64}` | **SelInv time (sec)**<br>`Measurement{Float64}` |
|------------------------:|-----------------:|-------------------:|-------------------------------------------------------:|------------------------------------------------:|
| bcsstk14                | 1806             | 63454              | 0.00334±0.00036                                        | 0.00327±0.00052                                 |
| bcsstk24                | 3562             | 159910             | 0.0068±0.0013                                          | 0.00699±0.00098                                 |
| bcsstk28                | 4410             | 219024             | 0.0092±0.0013                                          | 0.00852±0.00082                                 |
| bcsstk18                | 11948            | 149090             | 0.0201±0.0019                                          | 0.0313±0.0032                                   |
| bodyy6                  | 19366            | 134208             | 0.0186±0.002                                           | 0.0351±0.0023                                   |
| crystm03                | 24696            | 583770             | 0.076±0.014                                            | 0.157±0.015                                     |
| wathen120               | 36441            | 565761             | 0.0473±0.0021                                          | 0.0696±0.0067                                   |
| shipsec1                | 140874           | 3568176            | 1.043±0.018                                            | 1.774±0.083                                     |
| pwtk                    | 217918           | 11524432           | 1.144±0.044                                            | 1.881±0.032                                     |
| parabolic\_fem          | 525825           | 3674625            | 1.04±0.26                                              | 5.25458±NaN                                     |
| tmt\_sym                | 726713           | 5080961            | 1.7±0.11                                               | 7.34492±NaN                                     |
| ecology2                | 999999           | 4995991            | 1.35±0.17                                              | 12.1524±NaN                                     |
| G3\_circuit             | 1585478          | 7660826            | 4.15±0.5                                               | 48.0917±NaN                                     |

## Contributing

Check our [contribution guidelines](./CONTRIBUTING.md).

## References

[1] Lin, L., Yang, C., Meza, J. C., Lu, J., Ying, L., & E, W. (2011). SelInv---An Algorithm for Selected Inversion of a Sparse Symmetric Matrix. *ACM Transactions on Mathematical Software (TOMS)*, 37(4), 1-19.

[2] Erisman, A. M., & Tinney, W. F. (1975). On computing certain elements of the inverse of a sparse matrix. *Communications of the ACM*, 18(3), 177-179.

[3] Takahashi, K. (1973). Formation of sparse bus impedance matrix and its application to short circuit study. In *Proc. PICA Conference*, June, 1973.
