export selinv, selinv_diag

"""
    selinv(A::SparseMatrixCSC; depermute=false)
        -> @NamedTuple{Z::AbstractMatrix, p::Vector{Int64}}

Compute the selected inverse `Z` of `A`.
The sparsity pattern of `Z` corresponds to that of the Cholesky factor of `A`,
and the nonzero entries of `Z` match the corresponding entries in `A⁻¹`.

# Arguments
- `A::SparseMatrixCSC`: The sparse symmetric positive definite matrix for which
                        the selected inverse will be computed.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the selected inverse or not.

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector
of the corresponding sparse Cholesky factorization.
"""
selinv(A::SparseMatrixCSC; depermute = false) = selinv(cholesky(A); depermute = depermute)

"""
    selinv(F::SparseArrays.CHOLMOD.Factor; depermute=false)
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

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector
of the corresponding sparse Cholesky factorization.
"""
function selinv(F::SparseArrays.CHOLMOD.Factor; depermute = false)
    s = unsafe_load(pointer(F))
    if Bool(s.is_super)
        return selinv_supernodal(F; depermute = depermute)
    else
        return selinv_simplicial(F; depermute = depermute)
    end
end

"""
    selinv_diag(A::SparseMatrixCSC; depermute=true) -> Vector{Float64}

Compute the diagonal of the selected inverse of `A`.

This function efficiently computes only the diagonal entries of the selected inverse,
which correspond to the diagonal entries of `A⁻¹`. This is particularly useful for
applications like Gaussian Markov Random Fields where the diagonal of the inverse
precision matrix gives the marginal variances.

The implementation is optimized to avoid expensive sparse matrix permutation operations
by computing the permuted selected inverse first, extracting its diagonal, and then
applying the inverse permutation to the diagonal vector when needed.

# Arguments
- `A::SparseMatrixCSC`: The sparse symmetric positive definite matrix.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the diagonal or not. Default is `true`.

# Returns
A vector containing the diagonal entries of the selected inverse.
"""
selinv_diag(A::SparseMatrixCSC; depermute = true) = selinv_diag(cholesky(A); depermute = depermute)

"""
    selinv_diag(F::SparseArrays.CHOLMOD.Factor; depermute=true) -> Vector{Float64}

Compute the diagonal of the selected inverse based on a sparse Cholesky factorization `F`.

# Arguments
- `F::SparseArrays.CHOLMOD.Factor`: Sparse Cholesky factorization of some matrix `A`.

# Keyword arguments
- `depermute::Bool`: Whether to depermute the diagonal or not. Default is `true`.

# Returns
A vector containing the diagonal entries of the selected inverse.
"""
function selinv_diag(F::SparseArrays.CHOLMOD.Factor; depermute = true)
    # Always compute permuted selected inverse first for efficiency
    Z, p = selinv(F; depermute = false)
    
    # Extract diagonal from permuted matrix
    d = diag(Z)
    
    # Apply inverse permutation if depermuting is requested
    if depermute
        return d[invperm(p)]
    else
        return d
    end
end
