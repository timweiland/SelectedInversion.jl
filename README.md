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

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

SelectedInversion.jl is not yet a registered Julia package.
Until it is, you can install it from this GitHub repository.
To do so:

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Launch the Julia REPL and type `] add https://github.com/timweiland/SelectedInversion.jl`. 

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

## Contributing

Check our [contribution guidelines](./CONTRIBUTING.md).
