# Time-to-first-X (TTFX) benchmark for SelectedInversion.jl
#
# This script measures first-call latency by spawning fresh Julia processes.
# Run with: julia benchmark/ttfx_benchmark.jl

const PROJECT_PATH = dirname(@__DIR__)

# Matrix construction code (must match what's in the precompile workload)
const MATRIX_SETUP = """
using SparseArrays, LinearAlgebra

# 1D Laplacian (tridiagonal) - gives simplicial factorization
laplacian_1d(n) = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))

# Deterministic sparse SPD matrix - gives supernodal factorization for n >= 200
function make_supernodal_spd(n, entries_per_row)
    I_idx = Int[]
    J_idx = Int[]
    V = Float64[]
    for i in 1:n
        for k in 0:(entries_per_row - 1)
            j = mod(i + k * 7 + k^2, n) + 1
            push!(I_idx, i)
            push!(J_idx, j)
            push!(V, 1.0 + mod(i * 3 + j * 5, 10) / 10.0)
        end
    end
    A = sparse(I_idx, J_idx, V, n, n)
    return A * A' + 5I
end

A_simpl = laplacian_1d(50)
A_super = make_supernodal_spd(200, 15)
"""

"""
Run Julia code in a fresh process and return the output.
"""
function run_fresh_julia(code::String; timeout = 120)
    cmd = `$(Base.julia_cmd()) --project=$PROJECT_PATH -e $code`
    return read(cmd, String)
end

"""
Build Julia code that times an expression and prints the result.
"""
function timed_code(matrix_var::String, expr::String, label::String)
    return """
    # Load package
    t_load = @elapsed using SelectedInversion

    $MATRIX_SETUP

    A = $matrix_var

    # First call
    t_first = @elapsed begin
        $expr
    end

    # Second call (should be fast - already compiled)
    t_second = @elapsed begin
        $expr
    end

    println("$label")
    println("  Package load: ", round(t_load, digits=3), " s")
    println("  First call:   ", round(t_first, digits=3), " s")
    println("  Second call:  ", round(t_second, digits=3), " s")
    """
end

function main()
    println("="^60)
    println("TTFX Benchmark for SelectedInversion.jl")
    println("="^60)
    println()

    # Test 1: selinv with supernodal factorization
    println("Running: selinv (supernodal)...")
    code = timed_code(
        "A_super",
        "F = cholesky(A); selinv(F)",
        "selinv (supernodal)"
    )
    output = run_fresh_julia(code)
    println(output)

    # Test 2: selinv with simplicial factorization
    println("Running: selinv (simplicial)...")
    code = timed_code(
        "A_simpl",
        "F = cholesky(A; perm=1:size(A, 1)); selinv(F)",
        "selinv (simplicial)"
    )
    output = run_fresh_julia(code)
    println(output)

    # Test 3: selinv_diag with supernodal
    println("Running: selinv_diag (supernodal)...")
    code = timed_code(
        "A_super",
        "selinv_diag(A)",
        "selinv_diag (supernodal)"
    )
    output = run_fresh_julia(code)
    println(output)

    # Test 4: selinv_diag with simplicial
    println("Running: selinv_diag (simplicial)...")
    code = timed_code(
        "A_simpl",
        "F = cholesky(A; perm=1:size(A, 1)); selinv_diag(F)",
        "selinv_diag (simplicial)"
    )
    output = run_fresh_julia(code)
    println(output)

    println("="^60)
    println("Benchmark complete")
    return println("="^60)
end

main()
