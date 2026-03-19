# SupernodalMatrix API

For supernodal Cholesky factorizations, `SupernodalMatrix` stores the output of `selinv`.
This subtype of `AbstractMatrix` allows for a more efficient memory access
tailored to supernodal representations.

While you can just use it like a regular `AbstractMatrix` (i.e. you can get its size
and index into it as you would expect), you might be interested in more specialized
methods.

## Fields and construction
```@docs
SupernodalMatrix
SupernodalMatrix(F::SparseArrays.CHOLMOD.Factor; kwargs...)
```

## Methods
```@docs
val_range
col_range
get_rows
get_row_col_idcs
get_max_sup_size
```

```@docs
get_Sj
partition_Sj
```

```@docs
get_chunk
get_split_chunk
```
