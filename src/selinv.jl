export selinv, selinv_diag

"""
    selinv(A::SparseMatrixCSC; depermute=false, parallel=false)
        -> @NamedTuple{Z::AbstractMatrix, p::Vector{Int64}}

Compute the selected inverse `Z` of `A`.
The sparsity pattern of `Z` corresponds to that of the Cholesky factor of `A`,
and the nonzero entries of `Z` match the corresponding entries in `A⁻¹`.

# Arguments
- `A::SparseMatrixCSC`: The sparse symmetric positive definite matrix for which
                        the selected inverse will be computed.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the selected inverse or not.
- `parallel::Union{Bool, ParallelConfig}`: Enable parallel execution for supernodal
  factorizations. Has no effect on simplicial factorizations.
  - `false`: Sequential execution (default)
  - `true`: Parallel with default configuration
  - `ParallelConfig(...)`: Custom parallel configuration

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector
of the corresponding sparse Cholesky factorization.
"""
function selinv(A::SparseMatrixCSC; depermute = false, parallel = false)
    return selinv(cholesky(A); depermute = depermute, parallel = parallel)
end

"""
    selinv(F::SparseArrays.CHOLMOD.Factor; depermute=false, parallel=false)
        -> @NamedTuple{Z::AbstractMatrix, p::Vector{Int64}}

Compute the selected inverse `Z` of some matrix `A` based on its sparse Cholesky
factorization `F`.

# Arguments
- `F::SparseArrays.CHOLMOD.Factor`:
        Sparse Cholesky factorization of some matrix `A`.
        `F` will be used internally for the computations underlying the
        selected inversion of `A`.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the selected inverse or not.
- `parallel::Union{Bool, ParallelConfig}`: Enable parallel execution for supernodal
  factorizations. Has no effect on simplicial factorizations.
  - `false`: Sequential execution (default)
  - `true`: Parallel with default configuration
  - `ParallelConfig(...)`: Custom parallel configuration

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector
of the corresponding sparse Cholesky factorization.
"""
function selinv(F::SparseArrays.CHOLMOD.Factor; depermute = false, parallel = false)
    s = unsafe_load(pointer(F))
    if Bool(s.is_super)
        return selinv_supernodal(F; depermute = depermute, parallel = parallel)
    else
        return selinv_simplicial(F; depermute = depermute)
    end
end

"""
    selinv_diag(A; depermute=true, parallel=false) -> Vector{Float64}

Compute the diagonal of the selected inverse of `A`.

# Arguments
- `A`: A sparse matrix or a factorization object.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the diagonal or not. Default is `true`.
- `parallel::Union{Bool, ParallelConfig}`: Enable parallel execution for supernodal
  factorizations.

# Returns
A vector containing the diagonal entries of the selected inverse.
"""
function selinv_diag(A; depermute = true, parallel = false)
    # Always compute permuted selected inverse first for efficiency
    Z, p = selinv(A; depermute = false, parallel = parallel)

    # Extract diagonal from permuted matrix
    d = diag(Z)

    # Apply inverse permutation if depermuting is requested
    if depermute
        return d[invperm(p)]
    else
        return d
    end
end
