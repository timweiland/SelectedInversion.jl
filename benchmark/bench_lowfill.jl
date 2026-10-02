# Selected inversion on low-fill problems (see lowfill_problems.jl), compared to
# the Cholesky factorization it starts from.
#
# Usage: julia --project=benchmark benchmark/bench_lowfill.jl [blas_threads]

using BenchmarkTools
using SelectedInversion
using LinearAlgebra, SparseArrays

include("lowfill_problems.jl")

BLAS.set_num_threads(parse(Int, get(ARGS, 1, "1")))

fmt(t) = t < 1.0e-3 ? "$(round(t * 1.0e6; sigdigits = 3)) µs" : "$(round(t * 1.0e3; sigdigits = 3)) ms"

println("BLAS threads: $(BLAS.get_num_threads())")
println(
    rpad("problem", 14), rpad("n", 10), rpad("nnz(L)/n", 10), rpad("factor", 8),
    rpad("cholesky", 12), rpad("selinv", 12), rpad("selinv_diag", 13), "selinv/cholesky",
)
for (name, make) in lowfill_problems()
    A = make()
    F = cholesky(A)
    factor = Bool(unsafe_load(pointer(F)).is_super) ? "super" : "simpl"
    nnz_L_per_col = round(nnz(sparse(F.L)) / size(A, 1); sigdigits = 3)

    t_chol = @belapsed cholesky($A) seconds = 2
    t_sel = @belapsed selinv($F) seconds = 2
    t_diag = @belapsed selinv_diag($F) seconds = 2
    println(
        rpad(name, 14), rpad(size(A, 1), 10), rpad(nnz_L_per_col, 10), rpad(factor, 8),
        rpad(fmt(t_chol), 12), rpad(fmt(t_sel), 12), rpad(fmt(t_diag), 13),
        round(t_sel / t_chol; sigdigits = 3),
    )
end
