using Test, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Tests"
    @safetestset "InpParser Tests" begin
        include("inp_parser/parser.jl")
    end
    @safetestset "TopOptProblems Tests" begin
        include("topopt_problems/problems.jl")
        include("topopt_problems/metadata.jl")
    end
    @safetestset "Functions" begin
        include("Functions/test_common_fns.jl")
        include("Functions/test_buckling_fns.jl")
    end
    @safetestset "Solver" begin
        include("fea/solvers.jl")
    end
    @safetestset "Truss Problem Tests" begin
        include("truss_topopt_problems/test_problem.jl")
        include("truss_topopt_problems/test_fea.jl")
        include("truss_topopt_problems/test_buckling.jl")
        include("truss_topopt_problems/test_buckling_optimize.jl")
    end
end

if GROUP == "All" || GROUP == "Examples"
    @safetestset "CSIMP example" begin
        include("examples/csimp.jl")
    end
    @safetestset "Global stress example" begin
        include("examples/global_stress.jl")
    end
    @safetestset "Local stress example" begin
        include("examples/local_stress.jl")
    end
    @safetestset "More examples" begin
        include("examples/test_examples.jl")
    end
    @safetestset "Neural network example" begin
        include("examples/neural.jl")
    end
    @safetestset "Integer nonlinear programming for truss optimization example" begin
        include("examples/mixed_integer_truss/truss_compliance_2d1.jl")
    end
end

if GROUP == "All" || GROUP == "WCSMO14"
    # This was originlly part of https://github.com/JuliaTopOpt/TopOpt.jl_WCSMO21
    @safetestset "Continuum demos" begin
        include("wcsmo14/demos/continuum/cont_compliance1.jl")
        include("wcsmo14/demos/continuum/cont_compliance2.jl")
        include("wcsmo14/demos/continuum/cont_stress.jl")
    end
    @safetestset "Truss 2d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_2d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_2d2.jl")
    end
    @safetestset "Truss 3d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_3d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_3d2.jl")
    end
    @safetestset "WCSMO Benchmarks" begin
        include("wcsmo14/jl_benchmarks/compare_neo99_2D.jl")
        include("wcsmo14/jl_benchmarks/compare_polytop.jl")
        #include("wcsmo14/jl_benchmarks/compare_top3d.jl")
        #include("wcsmo14/jl_benchmarks/compare_top3d125.jl")
        #include("wcsmo14/jl_benchmarks/new_problems.jl")
    end
end
