using SelectedInversion
using Test
using Aqua

@testset "SelectedInversion.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SelectedInversion)
    end
    # Write your tests here.
end
