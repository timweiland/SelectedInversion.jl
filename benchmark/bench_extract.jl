# Benchmark for `selinv_extract` / `selinv_extract!`.
#
# Compares ways to read the selected inverse Σ at a small observation-style
# pattern B (≈ pattern(AᵀA)), the workload that motivated the primitive:
#
#   1. materialize   : Σ = sparse(S), then mask to B's pattern
#   2. getindex      : per-entry S[i, j] over B's pattern
#   3. extract       : selinv_extract(S, B)            (allocating)
#   4. extract!+plan : selinv_extract_setup once, then selinv_extract!(dest, S, plan)
#
# Run with:  julia --project=. benchmark/bench_extract.jl [k] [n_obs_frac]

using SelectedInversion
using LinearAlgebra, SparseArrays, Random

# 5-point Laplacian on a k×k grid (n = k²), a stand-in for a 2D SPDE factor.
function laplacian_2d(k)
    n = k * k
    idx(a, b) = (b - 1) * k + a
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    for b in 1:k, a in 1:k
        c = idx(a, b)
        push!(I_idx, c)
        push!(J_idx, c)
        push!(V, 4.0)
        for (da, db) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            a2, b2 = a + da, b + db
            if 1 <= a2 <= k && 1 <= b2 <= k
                push!(I_idx, c)
                push!(J_idx, idx(a2, b2))
                push!(V, -1.0)
            end
        end
    end
    return sparse(I_idx, J_idx, V, n, n) + 0.01I
end

# Mask Σ to B's pattern via ordinary indexing (the "materialize + use" tail).
function mask_to_pattern(Σ::SparseMatrixCSC, B::SparseMatrixCSC)
    dest = SparseMatrixCSC(
        B.m, B.n, copy(B.colptr), copy(rowvals(B)), zeros(Float64, nnz(B)),
    )
    rv = rowvals(B)
    nz = nonzeros(dest)
    @inbounds for j in 1:B.n, t in nzrange(B, j)
        nz[t] = Σ[rv[t], j]
    end
    return dest
end

function getindex_to_pattern(S, B::SparseMatrixCSC)
    dest = SparseMatrixCSC(
        B.m, B.n, copy(B.colptr), copy(rowvals(B)), zeros(Float64, nnz(B)),
    )
    rv = rowvals(B)
    nz = nonzeros(dest)
    @inbounds for j in 1:B.n, t in nzrange(B, j)
        nz[t] = S[rv[t], j]
    end
    return dest
end

# minimum wall time (s) and bytes allocated over `n` timed runs after warmup.
function bench(f, n)
    f()  # warmup / compile
    best = Inf
    bytes = typemax(Int)
    for _ in 1:n
        stats = @timed f()
        best = min(best, stats.time)
        bytes = min(bytes, stats.bytes)
    end
    return best, bytes
end

k = parse(Int, get(ARGS, 1, "120"))                 # n = k^2 (120 -> 14400)
n_obs_frac = parse(Float64, get(ARGS, 2, "0.25"))   # #observations / n

A = laplacian_2d(k)
n = size(A, 1)
F = cholesky(A)
is_super = Bool(unsafe_load(pointer(F)).is_super)

# Observation operator: each observation reads a local 3×3 patch of the k×k grid
# (a spatially-local pattern, as in a GMRF). The resulting B = AᵀA is sparse and
# a small subset of the factor fill — the regime selinv_extract targets.
rng = MersenneTwister(0)
m = max(1, round(Int, n_obs_frac * n))
I_obs = Int[]
J_obs = Int[]
V_obs = Float64[]
node(a, b) = (b - 1) * k + a
for o in 1:m
    a0 = rand(rng, 1:k)
    b0 = rand(rng, 1:k)
    for da in -1:1, db in -1:1
        a, b = a0 + da, b0 + db
        if 1 <= a <= k && 1 <= b <= k
            push!(I_obs, o)
            push!(J_obs, node(a, b))
            push!(V_obs, rand(rng))
        end
    end
end
Aobs = sparse(I_obs, J_obs, V_obs, m, n)
B = dropzeros(Aobs' * Aobs)

S = selinv(F; depermute = true).Z
fill_nnz = nnz(sparse(S))

println(
    "n = $n   supernodal = $is_super   nnz(factor fill) ≈ $fill_nnz   " *
        "nnz(B) = $(nnz(B))   (B/fill = $(round(nnz(B) / fill_nnz, digits = 3)))",
)
println("="^70)

# Correctness sanity: all methods agree.
Σ = sparse(S)
ref = mask_to_pattern(Σ, B)
@assert nonzeros(selinv_extract(S, B)) ≈ nonzeros(ref)
@assert nonzeros(getindex_to_pattern(S, B)) ≈ nonzeros(ref)

plan = selinv_extract_setup(S, B)
dest = selinv_extract(S, B)

reps = 30
t1, b1 = bench(() -> mask_to_pattern(sparse(S), B), reps)
t2, b2 = bench(() -> getindex_to_pattern(S, B), reps)
t3, b3 = bench(() -> selinv_extract(S, B), reps)
t4, b4 = bench(() -> selinv_extract!(dest, S, plan), reps)
tsetup, _ = bench(() -> selinv_extract_setup(S, B), 5)

fmt(t) = string(round(t * 1e3, digits = 3), " ms")
fmtb(b) = string(round(b / 1024, digits = 1), " KiB")

println(rpad("method", 26), rpad("min time", 14), "allocated")
println(rpad("1. sparse(S) + mask", 26), rpad(fmt(t1), 14), fmtb(b1))
println(rpad("2. getindex over S", 26), rpad(fmt(t2), 14), fmtb(b2))
println(rpad("3. selinv_extract", 26), rpad(fmt(t3), 14), fmtb(b3))
println(rpad("4. extract! + plan", 26), rpad(fmt(t4), 14), fmtb(b4))
println()
println("plan setup (one-off): ", fmt(tsetup))
println("speedup extract!+plan vs materialize: ", round(t1 / t4, digits = 1), "x")
println("speedup selinv_extract vs materialize: ", round(t1 / t3, digits = 1), "x")
