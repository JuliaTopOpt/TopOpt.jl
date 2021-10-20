using Test, SafeTestsets

# @safetestset "InpParser Tests" begin
#     include("inp_parser/parser.jl")
# end
# @safetestset "TopOptProblems Tests" begin
#     include("topopt_problems/problems.jl")
#     include("topopt_problems/metadata.jl")
# end
# @safetestset "Functions" begin
#     include("functions/test_common_fns.jl")
#     include("functions/test_buckling_fns.jl")
# end
# @safetestset "Solver" begin
#     include("fea/solvers.jl")
# end
@safetestset "Truss Problem Tests" begin
    # include("truss_topopt_problems/test_problem.jl")
    # include("truss_topopt_problems/test_fea.jl")
    include("truss_topopt_problems/test_buckling_optimize.jl")
end
# @safetestset "Examples" begin
#     @safetestset "CSIMP" begin
#         include("examples/csimp.jl")
#     end
#     @safetestset "Global Stress" begin
#         include("examples/global_stress.jl")
#     end
#     @safetestset "Local Stress" begin
#         include("examples/local_stress.jl")
#     end
#     @safetestset "More examples" begin 
#         include("examples/test_examples.jl")
#     end
# end

# @safetestset "CSIMP Tests" begin include("csimp.jl") end