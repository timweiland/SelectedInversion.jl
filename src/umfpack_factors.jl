# Read the numeric factors of a UMFPACK LU factorization into reusable buffers.
#
# `SparseArrays` exposes the factors through `F.L`, `F.U`, `F.p`, `F.q` and
# `F.Rs`, but each access allocates new arrays and `F.Rs` does not report how
# UMFPACK applied its row scaling: UMFPACK multiplies row `i` by `Rs[i]` only
# when every scale factor is large enough (`do_recip`); otherwise it divides by
# `Rs[i]`. The documented identity `F.L * F.U == (F.Rs .* A)[F.p, F.q]` therefore
# fails for matrices with very small row sums. This adapter calls
# `umfpack_*_get_numeric` directly, reading the factors in UMFPACK's native
# layout (L by rows, U by columns) together with `do_recip`, into the reusable
# buffers of a `_LUFactors` (see `lu_factors.jl`) so that repeated reads do not
# allocate.

const _UmfpackLU = SparseArrays.UMFPACK.UmfpackLU
const _LSS = SparseArrays.LibSuiteSparse

for (Tv, Ti, lunz, getnum) in (
        (:Float64, :Int64, :umfpack_dl_get_lunz, :umfpack_dl_get_numeric),
        (:Float64, :Int32, :umfpack_di_get_lunz, :umfpack_di_get_numeric),
        (:ComplexF64, :Int64, :umfpack_zl_get_lunz, :umfpack_zl_get_numeric),
        (:ComplexF64, :Int32, :umfpack_zi_get_lunz, :umfpack_zi_get_numeric),
    )
    isdefined(_LSS, getnum) || continue
    complex = Tv === :ComplexF64
    # Complex factors are read in UMFPACK's packed format: the imaginary-part
    # pointers are C_NULL and the real arrays hold interleaved values.
    call = if complex
        :(
            _LSS.$getnum(
                f.Lp, f.Lj, Ptr{Cdouble}(pointer(f.Lx)), C_NULL,
                f.Up, f.Ui, Ptr{Cdouble}(pointer(f.Ux)), C_NULL,
                f.P, f.Q, C_NULL, C_NULL, f.do_recip, f.Rs, F.numeric,
            )
        )
    else
        :(
            _LSS.$getnum(
                f.Lp, f.Lj, f.Lx, f.Up, f.Ui, f.Ux,
                f.P, f.Q, C_NULL, f.do_recip, f.Rs, F.numeric,
            )
        )
    end
    @eval function _read_factors!(f::_LUFactors{$Tv, $Ti}, F::_UmfpackLU{$Tv, $Ti})
        n = _check_lu_factorization(F)
        SparseArrays.UMFPACK.umfpack_numeric!(F)
        @lock F.lock begin
            status = _LSS.$lunz(f.lnz, f.unz, f.nrow, f.ncol, f.nzud, F.numeric)
            status == 0 || error("UMFPACK get_lunz failed with status $status")
            lnz, unz = Int(f.lnz[]), Int(f.unz[])
            f.n = n
            resize!(f.Lp, n + 1)
            resize!(f.Lj, lnz)
            resize!(f.Lx, lnz)
            resize!(f.Up, n + 1)
            resize!(f.Ui, unz)
            resize!(f.Ux, unz)
            resize!(f.P, n)
            resize!(f.Q, n)
            resize!(f.Rs, n)
            status = GC.@preserve f $call
            status == 0 || error("UMFPACK get_numeric failed with status $status")
        end
        return f
    end
end
