using SelectedInversion
using Test
using Aqua

include("utils.jl")
include("test_spd_matrix_collection.jl")

@testset "SelectedInversion.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SelectedInversion)
    end
    # Write your tests here.
end
