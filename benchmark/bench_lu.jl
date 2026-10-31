# Benchmark for selected inversion from a pivoted sparse LU factorization.
#
# Two augmented-system families whose local pivots can be zero or indefinite:
#
#   arrow : one cell coupling all effects, first prior curvature zero
#   cells : overlapping cells of three neighbouring effects, mixed-sign curvature
#
# For each factorization (UMFPACK `lu`, KLU `klu`) and size it reports the factor
# entries, the computed (closure) entries, and the minimum over repeated runs of
# the time and Julia-side allocation of
#
#   one-shot : selinv(F, B)                 (symbolic preparation + numeric)
#   reuse    : selinv!(dest, ws, F)         (numeric phase only)
#   refactor : lu!(F, K) or klu!(F, K)      (the solver's refactorization)
#
# UMFPACK's own factorization of the arrow family grows faster than linearly
# (its time dominates at the largest sizes); KLU's and the selected inversion's
# do not.
#
# Run with:  julia --project=. benchmark/bench_lu.jl

using SelectedInversion
using KLU
using LinearAlgebra, SparseArrays, Random

function augmented(D, B, W)
    r = size(B, 1)
    E = spdiagm(0 => ones(r))
    C = B / sqrt(2)
    return [D C' C'; C E + W / 2 -W / 2; C -W / 2 -E + W / 2]
end

function arrow(n)
    D = spdiagm(0 => [0.0; ones(n - 1)])
    B = sparse(fill(1, n), 1:n, [1.0; fill(0.01, n - 1)], 1, n)
    return augmented(D, B, sparse([1.0;;]))
end

function cells(n; rng = Xoshiro(1), width = 3)
    r = n - width + 1
    B = sparse(repeat(1:r, inner = width), vec([c + t for t in 0:(width - 1), c in 1:r]), randn(rng, r * width), r, n)
    D = spdiagm(0 => rand(rng, (0.0, 0.0, 1.0, -0.5), n))
    W = spdiagm(0 => rand(rng, (2.0, 1.0, 0.5, -0.25), r))
    return augmented(D, B, W)
end

# Request the structural entries of K and every diagonal entry.
request(K) = SparseMatrixCSC(size(K)..., K.colptr, rowvals(K), ones(Bool, nnz(K))) .| sparse(I, size(K)...)

function best(f, reps)
    f()
    runs = [@timed(f()) for _ in 1:reps]
    return minimum(r.time for r in runs), minimum(r.bytes for r in runs)
end

refactor!(F, K) = F isa KLU.KLUFactorization ? klu!(F, K) : lu!(F, K)

function bench(family, factorize, n; reps = 5)
    K = family(n)
    B = request(K)
    F = factorize(K)
    ws = LUSelInvWorkspace(F, B)
    dest = float(copy(B))
    t_one, b_one = best(() -> selinv(F, B), reps)
    t_re, b_re = best(() -> selinv!(dest, ws, F), reps)
    t_fac, _ = best(() -> refactor!(F, K), reps)
    computed = length(ws.lo_rowval) + length(ws.up_colval) + ws.n
    return (;
        family = nameof(family), factorize = nameof(factorize), n, size = size(K, 1),
        computed, requested = nnz(B), oneshot_s = t_one, oneshot_bytes = b_one,
        reuse_s = t_re, reuse_bytes = b_re, refactor_s = t_fac,
    )
end

for family in (arrow, cells), factorize in (lu, klu), n in (512, 4096, 32768)
    println(bench(family, factorize, n))
    flush(stdout)
end
