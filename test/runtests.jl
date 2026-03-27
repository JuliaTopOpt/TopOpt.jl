using Test, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core_Tests_1"
    @safetestset "InpParser Tests" begin
        include("inp_parser/parser.jl")
    end
    @safetestset "TopOptProblems Tests" begin
        include("topopt_problems/problems.jl")
        include("topopt_problems/metadata.jl")
        include("topopt_problems/test_io.jl")
    end
    @safetestset "Functions" begin
        include("Functions/test_common_fns.jl")
        include("Functions/test_buckling_fns.jl")
        include("Functions/test_truss_stress_fns.jl")
        include("Functions/test_mean_compliance.jl")
        include("Functions/test_thermal_compliance.jl")
        include("Functions/test_interpolation.jl")
        include("Functions/test_neural.jl")
    end
end

if GROUP == "All" || GROUP == "Core_Tests_2"
    @safetestset "Solver" begin
        include("FEA/solvers.jl")
        include("FEA/test_convergence.jl")
    end
    @safetestset "Utilities" begin
        include("Utilities/test_utils.jl")
        include("Utilities/test_penalties.jl")
    end
    @safetestset "CheqFilters" begin
        include("CheqFilters/test_filters.jl")
    end
    @safetestset "Truss Problem" begin
        include("truss_topopt_problems/test_problem.jl")
        include("truss_topopt_problems/test_fea.jl")
        include("truss_topopt_problems/test_buckling.jl")
        include("truss_topopt_problems/test_buckling_optimize.jl")
    end
    @safetestset "Integration" begin
        include("integration/test_end_to_end.jl")
    end
end

if GROUP == "All" || GROUP == "Core_Tests_3"
    @safetestset "BESO" begin
        include("Algorithms/test_beso.jl")
    end
end

if GROUP == "All" || GROUP == "Core_Tests_4"
    @safetestset "GESO" begin
        include("Algorithms/test_geso.jl")
    end
end

if GROUP == "All" || GROUP == "Examples_1"
    @safetestset "CSIMP example" begin
        include("examples/csimp.jl")
    end
end

if GROUP == "All" || GROUP == "Examples_2"
    @safetestset "Global stress example" begin
        include("examples/global_stress.jl")
    end
end

if GROUP == "All" || GROUP == "Examples_3"
    @safetestset "Local stress example" begin
        include("examples/local_stress.jl")
    end
end

if GROUP == "All" || GROUP == "Examples_4"
    @safetestset "More examples" begin
        include("examples/test_examples.jl")
    end
    @safetestset "Neural network example" begin
        include("examples/neural.jl")
    end
    @safetestset "Integer nonlinear programming for truss optimization example" begin
        include("examples/mixed_integer_truss/truss_compliance_2d1.jl")
    end
    @safetestset "Multi-material" begin
        include("examples/multimaterial.jl")
    end
    @safetestset "Heat sink example" begin
        include("examples/heat_sink.jl")
    end
end

if GROUP == "All" || GROUP == "WCSMO14_1"
    # This was originlly part of https://github.com/JuliaTopOpt/TopOpt.jl_WCSMO21
    @safetestset "Continuum demos" begin
        include("wcsmo14/demos/continuum/cont_compliance1.jl")
        # include("wcsmo14/demos/continuum/cont_compliance2.jl")
        # include("wcsmo14/demos/continuum/cont_stress.jl")
    end
end

if GROUP == "All" || GROUP == "WCSMO14_2"
    # This was originlly part of https://github.com/JuliaTopOpt/TopOpt.jl_WCSMO21
    @safetestset "Truss 2d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_2d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_2d2.jl")
    end
    @safetestset "Truss 3d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_3d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_3d2.jl")
    end
    @safetestset "WCSMO Benchmarks" begin
        # include("wcsmo14/jl_benchmarks/compare_neo99_2D.jl")
        # include("wcsmo14/jl_benchmarks/compare_polytop.jl")
        # include("wcsmo14/jl_benchmarks/compare_top3d.jl")
        # include("wcsmo14/jl_benchmarks/compare_top3d125.jl")
        # include("wcsmo14/jl_benchmarks/new_problems.jl")
    end
end