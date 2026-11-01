module SelectedInversionKLUExt

# Selected inversion from KLU factorizations. KLU factors a square matrix as
# `L * U + F == Diagonal(1 ./ Rs) * A[p, q]`: block triangular form with block
# diagonal `L` and `U`, the strictly block-upper part in `F`, and row scale
# factors `Rs` that divide the rows in permuted order. `_set_factors!` turns this
# into the `(R A)[P, Q] = L U` layout the LU selected inversion works on.

using SelectedInversion
using SelectedInversion: LUSelInvWorkspace, _LUFactors, _check_lu_factorization,
    _lu_workspace, _selinv_lu, _selinv_lu!, _selinv_lu_diag, _set_factors!
using KLU
using SparseArrays

const _KLUFactorization = KLU.AbstractKLUFactorization

SelectedInversion.LUSelInvWorkspace(K::_KLUFactorization{Tv, Ti}, B::SparseMatrixCSC) where {Tv, Ti} =
    _lu_workspace(K, B, Tv, Ti)

SelectedInversion.selinv(K::_KLUFactorization, B::SparseMatrixCSC) = _selinv_lu(K, B)

SelectedInversion.selinv!(dest::SparseMatrixCSC, ws::LUSelInvWorkspace, K::_KLUFactorization) =
    _selinv_lu!(dest, ws, K)

SelectedInversion.selinv_diag(K::_KLUFactorization; depermute = true) =
    _selinv_lu_diag(K, depermute)

# KLU.jl releases after 0.6.0 serialize calls on one factorization with
# `lock(K)`; hold it across all reads so that they see one factorization.
function _with_factor_lock(f, K::_KLUFactorization)
    hasmethod(lock, Tuple{typeof(K)}) || return f()
    lock(K)
    try
        return f()
    finally
        unlock(K)
    end
end

function SelectedInversion._read_factors!(
        f::_LUFactors{Tv, Ti}, K::_KLUFactorization{Tv, Ti},
    ) where {Tv, Ti}
    _check_lu_factorization(K)
    return _with_factor_lock(K) do
        _set_factors!(f, K.L, K.U, K.F, K.p, K.q, K.Rs)
    end
end

end
