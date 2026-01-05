module SelectedInversion

include("supernodal_matrix.jl")
include("elimination_tree.jl")
include("selinv_supernodal.jl")
include("selinv_simplicial.jl")
include("selinv.jl")
include("selinv_ldlt.jl")

using PrecompileTools

@setup_workload begin
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

    @compile_workload begin
        # Supernodal path
        F_super = cholesky(A_super)
        selinv(F_super)
        selinv(F_super; depermute = true)
        selinv_diag(A_super)
        selinv_diag(A_super; depermute = false)

        # Simplicial path (1D Laplacian with identity permutation)
        F_simpl = cholesky(A_simpl; perm = 1:size(A_simpl, 1))
        selinv(F_simpl)
        selinv(F_simpl; depermute = true)
    end
end

end
