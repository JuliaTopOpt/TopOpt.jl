using Test, SafeTestsets

@safetestset "InpParser Tests" begin
    include("InpParser/parser.jl")
end
@safetestset "TopOptProblems Tests" begin
    include("TopOptProblems/problems.jl")
    include("TopOptProblems/metadata.jl")
end
@safetestset "Truss Problem Tests" begin
    include("TrussTopOptProblems/test_problem.jl")
    include("TrussTopOptProblems/test_fea.jl")
end
@safetestset "Examples" begin
    @safetestset "CSIMP" begin
        include("examples/csimp.jl")
    end
    @safetestset "Global Stress" begin
        include("examples/stress.jl")
    end
    #@safetestset "More examples" begin
    #    include("examples/test_examples.jl")
    #end
end
