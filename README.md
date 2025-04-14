# SelectedInversion.jl

<p align="center">
<strong>🔥 Quickly compute selected entries of the inverse of a sparse matrix</strong>
</p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/SelectedInversion.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/SelectedInversion.jl/dev/)
[![Build Status](https://github.com/timweiland/SelectedInversion.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/SelectedInversion.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/SelectedInversion.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/SelectedInversion.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

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
Z = selinv(A) # its selected inverse
```

If you've already computed a Cholesky factorization of your matrix anyway,
you can directly pass it to `selinv` to save the factorization step.

```julia
C = cholesky(A)
Z = selinv(C)
```

## Contributing

Check our [contribution guidelines](./CONTRIBUTING.md).
