module SelectedInversionLDLFactorizations

using SelectedInversion
using LDLFactorizations
using LinearAlgebra

function SelectedInversion.selinv_simplicial(
    F::LDLFactorizations.LDLFactorization;
    depermute = false,
)
    Z = copy(F.L)
    for k in axes(Z, 2)
        Z[k, k] = 1 / F.D[k, k]
    end

    SelectedInversion._selinv_simplicial_Z!(Z)

    if depermute
        return (Z = Symmetric(Z, :L)[F.pinv, F.pinv], p = F.P)
    else
        return (Z = Symmetric(Z, :L), p = F.P)
    end
end

function SelectedInversion.selinv(F::LDLFactorizations.LDLFactorization; depermute = false)
    return SelectedInversion.selinv_simplicial(F; depermute = depermute)
end

function SelectedInversion.selinv_diag(F::LDLFactorizations.LDLFactorization; depermute = true)
    # Always compute permuted selected inverse first for efficiency
    Z, p = SelectedInversion.selinv(F; depermute = false)
    
    # Extract diagonal from permuted matrix
    d = LinearAlgebra.diag(Z)
    
    # Apply inverse permutation if depermuting is requested
    if depermute
        return d[invperm(p)]
    else
        return d
    end
end

end
