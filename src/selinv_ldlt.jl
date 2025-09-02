# Selected inversion for LDLt factorizations

"""
    selinv(F::LDLt; depermute=false) -> @NamedTuple{Z::AbstractMatrix, p::Vector{Int64}}

Compute the selected inverse based on Julia's built-in LDL^T factorization.

This method converts the LDL^T factorization to LL^T format by copying the L factor
and setting the diagonal to 1/D[k,k], then uses the existing simplicial selected 
inversion algorithm.

# Arguments
- `F::LDLt`: Built-in Julia LDL^T factorization (e.g., from `ldlt(A)` on SymTridiagonal)

# Keyword arguments
- `depermute::Bool`: Whether to depermute the selected inverse or not.

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector.
"""
function selinv(F::LDLt; depermute = false)
    # Convert LDL^T to LL^T format for existing algorithm
    # Copy L factor as sparse matrix and modify diagonal to be 1/D[k,k]
    Z = sparse(F.L)
    
    for k in axes(Z, 2)
        Z[k, k] = 1 / F.D[k, k]
    end
    
    # Use existing simplicial algorithm 
    _selinv_simplicial_Z!(Z)
    
    # No permutation for basic LDLt (identity permutation)
    p = collect(1:size(Z, 1))
    
    if depermute
        return (Z = Symmetric(Z, :L), p = p)
    else
        return (Z = Symmetric(Z, :L), p = p)
    end
end

"""
    selinv_diag(F::LDLt{T, SymTridiagonal{T, Vector{T}}}; depermute=true) -> Vector{Float64}

Optimized diagonal computation for LDLt factorizations of SymTridiagonal matrices.
Leverages precomputed D and L factors to skip the forward pass entirely.

# Arguments
- `F::LDLt`: LDLt factorization of a SymTridiagonal matrix

# Keyword arguments  
- `depermute::Bool`: Whether to depermute the diagonal or not. Default is `true`.

# Returns
A vector containing the diagonal entries of the selected inverse.
"""
function selinv_diag(F::LDLt{T, SymTridiagonal{T, Vector{T}}}; depermute = true) where T
    n = size(F, 1)
    
    # Extract precomputed factors - no forward pass needed!
    mod_diag = diag(F.D)  # This is our computed mod_diag from Kalman approach
    
    # Backward pass: compute diagonal of inverse (Kalman smoother-like)
    inv_diag = similar(mod_diag)
    inv_diag[n] = 1 / mod_diag[n]
    
    @inbounds for i in (n-1):-1:1
        # F.L[i+1, i] is exactly e[i] / mod_diag[i], so L[i+1, i]^2 = (e[i] / mod_diag[i])^2
        l_elem = F.L[i+1, i]
        inv_diag[i] = 1 / mod_diag[i] + (l_elem^2) * inv_diag[i+1]
    end
    
    return inv_diag
end

"""
    selinv_diag(A::SymTridiagonal; depermute=true) -> Vector{Float64}

Compute the diagonal of the selected inverse of a symmetric tridiagonal matrix.

This method uses a specialized Kalman smoother-inspired algorithm that is much faster
than the general approach, achieving O(n) time complexity with only two passes through
the data and minimal allocations.

# Arguments
- `A::SymTridiagonal`: The symmetric tridiagonal matrix

# Keyword arguments
- `depermute::Bool`: Whether to depermute the diagonal or not. Default is `true`.
                     For SymTridiagonal matrices, no permutation is applied, so this
                     parameter has no effect but is kept for interface consistency.

# Returns
A vector containing the diagonal entries of the selected inverse.
"""
function selinv_diag(A::SymTridiagonal; depermute = true)
    n = size(A, 1)
    d = A.dv  # diagonal elements
    e = A.ev  # off-diagonal elements
    
    # Forward pass: compute modified diagonal elements (Kalman filter-like)
    # This corresponds to computing the diagonal of the Cholesky factor
    mod_diag = similar(d)
    mod_diag[1] = d[1]
    
    @inbounds for i in 2:n
        mod_diag[i] = d[i] - e[i-1]^2 / mod_diag[i-1]
    end
    
    # Backward pass: compute diagonal of inverse (Kalman smoother-like)
    # This efficiently computes the diagonal of A^{-1} using the forward pass results
    inv_diag = similar(d)
    inv_diag[n] = 1 / mod_diag[n]
    
    @inbounds for i in (n-1):-1:1
        inv_diag[i] = 1 / mod_diag[i] + (e[i]^2 / mod_diag[i]^2) * inv_diag[i+1]
    end
    
    return inv_diag
end

"""
    selinv(A::SymTridiagonal; depermute=false) -> @NamedTuple{Z::AbstractMatrix, p::Vector{Int64}}

Compute the selected inverse of a symmetric tridiagonal matrix.

This method leverages Julia's efficient built-in `ldlt` factorization for `SymTridiagonal` 
matrices, which has a specialized fast implementation for tridiagonal structures.

# Arguments
- `A::SymTridiagonal`: The symmetric tridiagonal matrix

# Keyword arguments
- `depermute::Bool`: Whether to depermute the selected inverse or not.

# Returns
A named tuple `Zp`.
`Zp.Z` is the selected inverse, and `Zp.p` is the permutation vector.
"""
selinv(A::SymTridiagonal; depermute = false) = selinv(ldlt(A); depermute = depermute)