using Test, SafeTestsets

@safetestset "InpParser Tests" begin include("InpParser/parser.jl") end
@safetestset "MMA Tests" begin include("MMA/mma.jl") end
@safetestset "TopOptProblems Tests" begin
    include("TopOptProblems/problems.jl")
    #include("TopOptProblems/metadata.jl")
end
@safetestset "AugLag Tests" begin
    include("AugLag/auglag.jl")
    include("AugLag/compliance.jl")
end
@safetestset "Global Stress Tests" begin include("stress.jl") end
@safetestset "Example Tests" begin include("test_examples.jl") end

@safetestset "Truss Problem Tests" begin
    include("TrussTopOptProblems/test_problem.jl")
    include("TrussTopOptProblems/test_fea.jl")
end

# @safetestset "CSIMP Tests" begin include("csimp.jl") end