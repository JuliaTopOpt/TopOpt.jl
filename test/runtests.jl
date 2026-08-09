using Test, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

# Check if we're running opposite preference tests
const OPPOSITE_PREFERENCE = occursin("Opposite_Preference", GROUP)

if OPPOSITE_PREFERENCE
    using TopOpt
    # Skip tests if preference is not set to false (i.e., still default/true)
    if TopOpt.PENALTY_BEFORE_INTERPOLATION != false
        @info "Skipping tests: PENALTY_BEFORE_INTERPOLATION is not false (current value: $(TopOpt.PENALTY_BEFORE_INTERPOLATION))"
        exit(0)  # Exit successfully but skip all tests
    else
        @info "Running tests with PENALTY_BEFORE_INTERPOLATION = false"
    end
end

# Strip the _Opposite_Preference suffix to get the actual test group
const ACTUAL_GROUP = replace(GROUP, "_Opposite_Preference" => "")

if ACTUAL_GROUP in ("All", "Core_Tests")
    @safetestset "Ferrite Upgrade Behavior" begin
        include("ferrite_upgrade_behavior.jl")
    end
    @safetestset "InpParser Tests" begin
        include("inp_parser/parser.jl")
        include("inp_parser/test_inpstiffness.jl")
    end
    @safetestset "TopOptProblems Tests" begin
        include("topopt_problems/problems.jl")
        include("topopt_problems/metadata.jl")
        include("topopt_problems/test_io.jl")
        include("topopt_problems/test_grids.jl")
        include("topopt_problems/test_assembly.jl")
        include("topopt_problems/test_show.jl")
        include("topopt_problems/element_stiffness_matrix.jl")
        include("topopt_problems/test_elementmatrix.jl")
        include("topopt_problems/test_assemble_functions.jl")
        include("topopt_problems/test_multiload.jl")
        include("topopt_problems/test_pressure.jl")
    end
    @safetestset "Functions" begin
        include("Functions/test_common_fns.jl")
        include("Functions/test_fixed_element.jl")
        include("Functions/test_buckling_fns.jl")
        include("Functions/test_truss_stress_fns.jl")
        include("Functions/test_mean_compliance.jl")
        include("Functions/test_thermal_compliance.jl")
        include("Functions/test_interpolation.jl")
        include("Functions/test_neural.jl")
        include("Functions/test_show.jl")
        include("Functions/test_function_utils.jl")
        include("Functions/test_trace.jl")
        include("Functions/test_block_compliance.jl")
        include("Functions/test_compute_mean_compliance_svd.jl")
        include("Functions/test_element_stress_tensor.jl")
        include("Functions/test_generate_scenarios.jl")
        include("Functions/test_getdim.jl")
        include("Functions/test_hadamard.jl")
        include("Functions/test_mean_compliance_branches.jl")
        include("Functions/test_stress_tensor_rrule.jl")
        include("Functions/test_truss_stress_rrule.jl")
    end
    @safetestset "Solver" begin
        include("FEA/solvers.jl")
        include("FEA/test_convergence.jl")
        include("FEA/test_simulate.jl")
        include("FEA/test_cg_energy_criteria.jl")
        include("FEA/test_operator.jl")
        include("FEA/misc.jl")
        include("FEA/test_cg_assembly_safe.jl")
        include("FEA/test_preconditioner.jl")
    end
    @safetestset "Utilities" begin
        include("Utilities/test_utils.jl")
        include("Utilities/test_penalties.jl")
        include("Utilities/test_show.jl")
    end
    @safetestset "CheqFilters" begin
        include("CheqFilters/test_filters.jl")
    end
    @safetestset "Truss Problem" begin
        include("truss_topopt_problems/test_problem.jl")
        include("truss_topopt_problems/test_fea.jl")
        include("truss_topopt_problems/test_buckling.jl")
        include("truss_topopt_problems/test_buckling_optimize.jl")
        include("truss_topopt_problems/test_simulate_truss.jl")
        include("truss_topopt_problems/utils.jl")
    end
    @safetestset "BESO" begin
        include("Algorithms/test_beso.jl")
        include("Algorithms/test_geso.jl")
    end
    @safetestset "Integration" begin
        include("integration/test_end_to_end.jl")
    end
end

const LITERATE_DIR = joinpath(@__DIR__, "../docs/src/literate")

if ACTUAL_GROUP in ("All", "Examples_1")
    @safetestset "CSIMP example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "csimp.jl"))
            end
        end
    end
end

if ACTUAL_GROUP in ("All", "Examples_2")
    @safetestset "Global stress example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "global_stress.jl"))
            end
        end
    end
end

if ACTUAL_GROUP in ("All", "Examples_3")
    @safetestset "Local stress example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "local_stress.jl"))
            end
        end
    end
end

if ACTUAL_GROUP in ("All", "Examples_4")
    @safetestset "SIMP example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "simp.jl"))
            end
        end
    end
    @safetestset "BESO example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "beso.jl"))
            end
        end
    end
    @safetestset "GESO example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "geso.jl"))
            end
        end
    end
    @safetestset "Neural network example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "neural.jl"))
            end
        end
    end
    @safetestset "Neural network (Adam) example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "neural2.jl"))
            end
        end
    end
    @safetestset "Integer nonlinear programming for truss optimization example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "mixed_integer_truss.jl"))
            end
        end
    end
    @safetestset "Multi-material" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "multimaterial.jl"))
            end
        end
    end
    @safetestset "Heat sink example" begin
        mktempdir() do dir
            cd(dir) do
                include(joinpath(Main.LITERATE_DIR, "heat_sink.jl"))
            end
        end
    end
end

if ACTUAL_GROUP in ("All", "WCSMO14_1")
    # This was originlly part of https://github.com/JuliaTopOpt/TopOpt.jl_WCSMO21
    @safetestset "Continuum demos" begin
        include("wcsmo14/demos/continuum/cont_compliance1.jl")
        # cont_compliance2.jl and cont_stress.jl are additional continuum demos
        # that are not included in regular CI testing to keep test times reasonable.
        # They can be run manually for extended validation.
        # include("wcsmo14/demos/continuum/cont_compliance2.jl")
        # include("wcsmo14/demos/continuum/cont_stress.jl")
    end
end

if ACTUAL_GROUP in ("All", "WCSMO14_2")
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
        # Benchmark comparison files are excluded from regular CI testing
        # as they are used for performance benchmarking against other topopt
        # implementations and can take significant time to run.
        # They can be run manually for benchmarking purposes.
        # include("wcsmo14/jl_benchmarks/compare_neo99_2D.jl")
        # include("wcsmo14/jl_benchmarks/compare_polytop.jl")
        # include("wcsmo14/jl_benchmarks/compare_top3d.jl")
        # include("wcsmo14/jl_benchmarks/compare_top3d125.jl")
        # include("wcsmo14/jl_benchmarks/new_problems.jl")
        # include("wcsmo14/jl_benchmarks/compare_topopt_py.jl")
    end
end
