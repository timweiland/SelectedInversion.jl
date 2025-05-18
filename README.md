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
| bcsstk14                | 1806             | 63454              | 0.00335±0.00033                                        | 0.00295±0.0008                                  |
| bcsstk24                | 3562             | 159910             | 0.00666±0.00055                                        | 0.0067±0.0014                                   |
| bcsstk28                | 4410             | 219024             | 0.00905±0.00076                                        | 0.0082±0.0017                                   |
| bcsstk18                | 11948            | 149090             | 0.0199±0.00073                                         | 0.0466±0.0064                                   |
| bodyy6                  | 19366            | 134208             | 0.0184±0.0021                                          | 0.042±0.011                                     |
| crystm03                | 24696            | 583770             | 0.075±0.016                                            | 0.296±0.029                                     |
| wathen120               | 36441            | 565761             | 0.0481±0.0057                                          | 0.087±0.017                                     |
| shipsec1                | 140874           | 3568176            | 2.075±0.058                                            | 7.70644±NaN                                     |
| pwtk                    | 217918           | 11524432           | 1.36±0.25                                              | 2.054±0.066                                     |
| parabolic\_fem          | 525825           | 3674625            | 1.002±0.066                                            | 6.63912±NaN                                     |
| tmt\_sym                | 726713           | 5080961            | 4.201±0.013                                            | 10.8521±NaN                                     |
| ecology2                | 999999           | 4995991            | 1.52±0.14                                              | 13.7855±NaN                                     |
| G3\_circuit             | 1585478          | 7660826            | 10.42±NaN                                              | 53.636±NaN                                      |

## Contributing

Check our [contribution guidelines](./CONTRIBUTING.md).

## References

[1] Lin, L., Yang, C., Meza, J. C., Lu, J., Ying, L., & E, W. (2011). SelInv---An Algorithm for Selected Inversion of a Sparse Symmetric Matrix. *ACM Transactions on Mathematical Software (TOMS)*, 37(4), 1-19.

[2] Erisman, A. M., & Tinney, W. F. (1975). On computing certain elements of the inverse of a sparse matrix. *Communications of the ACM*, 18(3), 177-179.

[3] Takahashi, K. (1973). Formation of sparse bus impedance matrix and its application to short circuit study. In *Proc. PICA Conference*, June, 1973.
