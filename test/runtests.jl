using Test, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

# Check if we're running opposite preference tests
const OPPOSITE_PREFERENCE = occursin("Opposite_Preference", GROUP)

if OPPOSITE_PREFERENCE
    using TopOpt
    # Skip tests if preference is not set to false (i.e., still default/true)
    if TopOpt.PENALTY_BEFORE_INTERPOLATION != false
        @info "Skipping tests: TopOpt.PENALTY_BEFORE_INTERPOLATION is not false (current value: $(TopOpt.PENALTY_BEFORE_INTERPOLATION))"
        exit(0)  # Exit successfully but skip all tests
    else
        @info "Running tests with TopOpt.PENALTY_BEFORE_INTERPOLATION = false"
    end
end

# Strip the _Opposite_Preference suffix to get the actual test group
const ACTUAL_GROUP = replace(GROUP, "_Opposite_Preference" => "")

if ACTUAL_GROUP in ("All", "Core_Tests_1")
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
        include("topopt_problems/test_visualize.jl")
    end
    @safetestset "Functions" begin
        include("Functions/test_common_fns.jl")
        include("Functions/test_fixed_element.jl")
        include("Functions/test_buckling_fns.jl")
        include("Functions/test_truss_stress_fns.jl")
        include("Functions/test_mean_compliance.jl")
        include("Functions/test_thermal_compliance.jl")
        include("Functions/test_temperature.jl")
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
        include("Functions/test_stress_relaxation.jl")
        include("Functions/test_truss_stress_rrule.jl")
    end
end

if ACTUAL_GROUP in ("All", "Core_Tests_2")
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

if ACTUAL_GROUP in ("All", "Aqua_Tests")
    @safetestset "Aqua" begin
        include("aqua.jl")
    end
end

# Tutorial tests are now run via Quarto render in CI (tutorials job)
# Each .qmd file is executed during rendering, catching any execution errors.
# No separate test runner needed - Quarto render serves as the test.

if ACTUAL_GROUP in ("All", "WCSMO14_1")
    @safetestset "Continuum demos" begin
        include("wcsmo14/demos/continuum/cont_compliance1.jl")
    end
end

if ACTUAL_GROUP in ("All", "WCSMO14_2")
    @safetestset "Truss 2d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_2d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_2d2.jl")
    end
    @safetestset "Truss 3d demos" begin
        include("wcsmo14/demos/truss/truss_compliance_3d1.jl")
        include("wcsmo14/demos/truss/truss_compliance_3d2.jl")
    end
end

if ACTUAL_GROUP in ("All", "OpenLSTO_Tests")
    @safetestset "OpenLSTO compliance reference" begin
        include("OpenLSTO/test_compliance_reference.jl")
    end
end

if ACTUAL_GROUP in ("All", "JET_Tests")
    @safetestset "JET" begin
        include("jet.jl")
    end
end
