using Test, SafeTestsets

@safetestset "InpParser Tests" begin
    include("inp_parser/parser.jl")
end
@safetestset "TopOptProblems Tests" begin
    include("topopt_problems/problems.jl")
    include("topopt_problems/metadata.jl")
end
@safetestset "Functions" begin
    include("functions.jl")
end
@safetestset "Truss Problem Tests" begin
    include("truss_topopt_problems/test_problem.jl")
    include("truss_topopt_problems/test_fea.jl")
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
