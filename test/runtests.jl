using Test

@testset "GREMLModels" begin
    include(joinpath(@__DIR__, "..", "tests", "fit.jl"))
end
