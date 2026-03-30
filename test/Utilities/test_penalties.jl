using TopOpt, Test, LinearAlgebra, Zygote, ForwardDiff
using TopOpt: PowerPenalty, RationalPenalty, SinhPenalty, HeavisideProjection, SigmoidProjection, ProjectedPenalty, PseudoDensities, setpenalty!, getpenalty, getprevpenalty
using TopOpt.FEA: HeatTransfer

@testset "PowerPenalty" begin
    # Test construction
    pp = PowerPenalty(3.0)
    @test pp.p == 3.0

    # Test application to scalar values
    @test pp(0.5) ≈ 0.5^3
    @test pp(0.0) ≈ 0.0
    @test pp(1.0) ≈ 1.0

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = pp(x)
    @test result.x ≈ [0.1^3, 0.5^3, 0.9^3]
    @test result isa PseudoDensities

    # Test derivative with ForwardDiff
    f(x) = PowerPenalty(3.0)(x)
    @test ForwardDiff.derivative(f, 0.5) ≈ 3.0 * 0.5^2
    @test ForwardDiff.derivative(f, 0.0) ≈ 0.0

    # Test derivative with Zygote
    g = x -> PowerPenalty(2.0)(x)
    @test Zygote.gradient(g, 0.5)[1] ≈ 2.0 * 0.5

    # Test copy
    pp_copy = copy(pp)
    @test pp_copy.p == pp.p
    pp_copy.p = 2.0
    @test pp.p == 3.0  # Original unchanged

    # Test setpenalty!
    pp2 = PowerPenalty(2.0)
    setpenalty!(pp2, 3.5)
    @test pp2.p == 3.5

    # Test different penalty values
    @test PowerPenalty(1.0)(0.5) ≈ 0.5
    @test PowerPenalty(2.0)(0.5) ≈ 0.25
    @test PowerPenalty(3.0)(0.5) ≈ 0.125
end

@testset "setpenalty! Error Handling" begin
    # Create a minimal FEASolver for testing
    # First create a simple problem
    nels = (2, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

    # Create a solver using FEASolver with DirectSolver
    solver = FEASolver(DirectSolver, problem)

    # Test that setpenalty! throws ArgumentError for unsupported types
    # Note: Numbers are valid (update penalty value), AbstractPenalty is valid (replace penalty)
    # Invalid types include: strings, arrays, dicts, etc.
    @test_throws ArgumentError setpenalty!(solver, "string")
    @test_throws ArgumentError setpenalty!(solver, [1.0, 2.0, 3.0])
end

@testset "setpenalty! on Compliance object" begin
    # Create a minimal problem and solver
    nels = (2, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

    # Create a solver using FEASolver with DirectSolver
    solver = FEASolver(DirectSolver, problem)
    
    # Create a Compliance object with the solver
    comp = Compliance(solver)

    # Test initial penalty value
    initial_penalty = getpenalty(comp)
    @test initial_penalty isa PowerPenalty
    @test initial_penalty.p == 1.0  # Default penalty value
    
    # Store the initial penalty value for comparison
    initial_p = initial_penalty.p

    # Test setpenalty! on Compliance object with a new penalty value
    new_p = 3.0
    setpenalty!(comp, new_p)
    
    # Verify the penalty was updated on the Compliance
    updated_penalty = getpenalty(comp)
    @test updated_penalty.p == new_p
    
    # Verify the penalty was also updated on the underlying solver
    @test getpenalty(solver).p == new_p
    
    # Verify prev_penalty on solver stores the old value
    @test getprevpenalty(solver).p == initial_p
end

@testset "RationalPenalty" begin
    # Test construction
    rp = RationalPenalty(3.0)
    @test rp.p == 3.0

    # Test application to scalar values
    x = 0.5
    @test rp(x) ≈ x / (1.0 + rp.p * (1.0 - x))
    @test rp(0.0) ≈ 0.0
    @test rp(1.0) ≈ 1.0

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = rp(x)
    expected = [rp(0.1), rp(0.5), rp(0.9)]
    @test result.x ≈ expected

    # Test derivative with ForwardDiff
    f(x) = RationalPenalty(3.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Analytical derivative: (1 + p) / (1 + p(1-x))^2
    expected_df = (1.0 + 3.0) / (1.0 + 3.0 * (1.0 - 0.5))^2
    @test df ≈ expected_df

    # Test copy
    rp_copy = copy(rp)
    @test rp_copy.p == rp.p

    # Test setpenalty!
    setpenalty!(rp, 4.0)
    @test rp.p == 4.0
end

@testset "SinhPenalty" begin
    # Test construction
    sp = SinhPenalty(3.0)
    @test sp.p == 3.0

    # Test application to scalar values
    x = 0.5
    # sinh_penalty = sinh(p*x) / sinh(p)
    @test sp(x) ≈ sinh(3.0 * x) / sinh(3.0)
    @test sp(0.0) ≈ 0.0
    @test isapprox(sp(1.0), 1.0, atol=1e-10)

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = sp(x)
    expected = [sp(0.1), sp(0.5), sp(0.9)]
    @test result.x ≈ expected

    # Test derivative with ForwardDiff
    f(x) = SinhPenalty(3.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Analytical derivative: p * cosh(p*x) / sinh(p)
    expected_df = 3.0 * cosh(3.0 * 0.5) / sinh(3.0)
    @test df ≈ expected_df

    # Test copy and setpenalty!
    sp_copy = copy(sp)
    @test sp_copy.p == sp.p
    setpenalty!(sp, 2.0)
    @test sp.p == 2.0
end

@testset "HeavisideProjection" begin
    # Test construction
    hp = HeavisideProjection(5.0)
    @test hp.β == 5.0

    # Test projection function
    x = 0.5
    # heaviside_proj = 1 - exp(-β*x) + x*exp(-β)
    expected = 1 - exp(-5.0 * x) + x * exp(-5.0)
    @test hp(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = hp(x_pd)
    @test result.x ≈ [hp(0.1), hp(0.5), hp(0.9)]

    # Test with AbstractArray
    x_arr = [0.1, 0.5, 0.9]
    result_arr = hp(x_arr)
    @test result_arr ≈ [hp(0.1), hp(0.5), hp(0.9)]

    # Test derivative with ForwardDiff
    f(x) = HeavisideProjection(5.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # At x = 0.5, derivative should be relatively large
    @test df > 0.0

    # Test at boundaries
    @test hp(0.0) < 0.01  # Should be close to 0
    @test hp(1.0) > 0.99  # Should be close to 1

    # Test copy
    hp_copy = copy(hp)
    @test hp_copy.β == hp.β

    # Test with different β values
    hp2 = HeavisideProjection(3.0)
    @test hp2.β == 3.0
    @test hp2(0.5) ≈ 1 - exp(-3.0 * 0.5) + 0.5 * exp(-3.0)
end

@testset "SigmoidProjection" begin
    # Test construction
    sp = SigmoidProjection(4.0)
    @test sp.β == 4.0

    # Test projection function
    x = 0.5
    # sigmoid_proj = 1 / (1 + exp((β+1)*(-x+0.5)))
    expected = 1.0 / (1.0 + exp((4.0 + 1) * (-x + 0.5)))
    @test sp(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = sp(x_pd)
    @test result.x ≈ [sp(0.1), sp(0.5), sp(0.9)]

    # Test with AbstractArray
    x_arr = [0.2, 0.5, 0.8]
    result_arr = sp(x_arr)
    @test result_arr ≈ [sp(0.2), sp(0.5), sp(0.8)]

    # Test derivative with ForwardDiff
    f(x) = SigmoidProjection(4.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Just verify derivative is positive and finite
    @test df > 0.0
    @test isfinite(df)

    # Test at boundaries (with β=4, sigmoid doesn't saturate completely)
    @test sp(0.0) < 0.1  # Should be relatively small
    @test sp(1.0) > 0.9  # Should be relatively large

    # Test copy
    sp_copy = copy(sp)
    @test sp_copy.β == sp.β

    # Test with different β values
    sp2 = SigmoidProjection(2.0)
    @test sp2.β == 2.0
    @test sp2(0.5) ≈ 1 / (1 + exp((2.0 + 1) * (-0.5 + 0.5)))
end

@testset "ProjectedPenalty" begin
    # Test construction
    pp = PowerPenalty(3.0)
    proj = HeavisideProjection(5.0)
    pp_proj = ProjectedPenalty(pp, proj)
    
    @test pp_proj.penalty == pp
    @test pp_proj.proj == proj

    # Test application: penalty(proj(x))
    x = 0.5
    expected = pp(proj(x))
    @test pp_proj(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = pp_proj(x_pd)
    expected_values = [pp(proj(0.1)), pp(proj(0.5)), pp(proj(0.9))]
    @test result.x ≈ expected_values

    # Test property forwarding
    @test pp_proj.p == 3.0  # Forwarded from PowerPenalty
    
    # Test copy
    pp_proj_copy = copy(pp_proj)
    @test pp_proj_copy.p == pp_proj.p
    @test pp_proj_copy.proj.β == proj.β

    # Test setpenalty!
    setpenalty!(pp_proj, 4.0)
    @test pp_proj.p == 4.0

    # Test with different penalty/projection combinations
    rp = RationalPenalty(2.0)
    sp = SigmoidProjection(3.0)
    rp_sp = ProjectedPenalty(rp, sp)
    
    x = PseudoDensities([0.2, 0.6, 0.8])
    result = rp_sp(x)
    expected = [rp(sp(0.2)), rp(sp(0.6)), rp(sp(0.8))]
    @test result.x ≈ expected

    # Test gradient
    f(x) = ProjectedPenalty(PowerPenalty(2.0), HeavisideProjection(3.0))(x)
    g = Zygote.gradient(f, 0.5)[1]
    @test g isa Real
end

using TopOpt.Utilities: get_ρ, get_ρ_dρ, density

@testset "get_ρ - penalized density computation" begin
    # Note: PENALTY_BEFORE_INTERPOLATION is a compile-time preference.
    # The tests below verify the function behaves correctly for the current configuration.
    # To test the other mode, restart Julia with the preference changed.
    @info "Testing get_ρ with PENALTY_BEFORE_INTERPOLATION = $(TopOpt.PENALTY_BEFORE_INTERPOLATION)"

    # Test with PowerPenalty
    @testset "PowerPenalty" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001

        # Test basic computation at x_e = 0.5
        x_e = 0.5
        result = get_ρ(x_e, penalty, xmin)

        # Manual computation based on current mode
        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            # density(penalty(x_e), xmin)
            expected = density(penalty(x_e), xmin)
        else
            # penalty(density(x_e, xmin))
            expected = penalty(density(x_e, xmin))
        end
        @test result ≈ expected

        # Test edge case: x_e = 0.0
        result_0 = get_ρ(0.0, penalty, xmin)
        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected_0 = density(penalty(0.0), xmin)
        else
            expected_0 = penalty(density(0.0, xmin))
        end
        @test result_0 ≈ expected_0

        # Test edge case: x_e = 1.0
        result_1 = get_ρ(1.0, penalty, xmin)
        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected_1 = density(penalty(1.0), xmin)
        else
            expected_1 = penalty(density(1.0, xmin))
        end
        @test result_1 ≈ expected_1

        # Test with different xmin values
        for xmin_test in [0.0, 0.001, 0.01, 0.1]
            result_xmin = get_ρ(0.5, penalty, xmin_test)
            if TopOpt.PENALTY_BEFORE_INTERPOLATION
                expected_xmin = density(penalty(0.5), xmin_test)
            else
                expected_xmin = penalty(density(0.5, xmin_test))
            end
            @test result_xmin ≈ expected_xmin
        end

        # Test with different penalty values
        for p in [1.0, 2.0, 3.0, 5.0]
            penalty_p = PowerPenalty(p)
            result_p = get_ρ(0.5, penalty_p, xmin)
            if TopOpt.PENALTY_BEFORE_INTERPOLATION
                expected_p = density(penalty_p(0.5), xmin)
            else
                expected_p = penalty_p(density(0.5, xmin))
            end
            @test result_p ≈ expected_p
        end
    end

    # Test with RationalPenalty
    @testset "RationalPenalty" begin
        penalty = RationalPenalty(3.0)
        xmin = 0.001

        x_e = 0.5
        result = get_ρ(x_e, penalty, xmin)

        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected = density(penalty(x_e), xmin)
        else
            expected = penalty(density(x_e, xmin))
        end
        @test result ≈ expected

        # Test edge cases
        @test get_ρ(0.0, penalty, xmin) ≈ (TopOpt.PENALTY_BEFORE_INTERPOLATION ? density(penalty(0.0), xmin) : penalty(density(0.0, xmin)))
        @test get_ρ(1.0, penalty, xmin) ≈ (TopOpt.PENALTY_BEFORE_INTERPOLATION ? density(penalty(1.0), xmin) : penalty(density(1.0, xmin)))
    end

    # Test with SinhPenalty
    @testset "SinhPenalty" begin
        penalty = SinhPenalty(3.0)
        xmin = 0.001

        x_e = 0.5
        result = get_ρ(x_e, penalty, xmin)

        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected = density(penalty(x_e), xmin)
        else
            expected = penalty(density(x_e, xmin))
        end
        @test result ≈ expected

        # Test edge cases
        @test get_ρ(0.0, penalty, xmin) ≈ (TopOpt.PENALTY_BEFORE_INTERPOLATION ? density(penalty(0.0), xmin) : penalty(density(0.0, xmin)))
        @test get_ρ(1.0, penalty, xmin) ≈ (TopOpt.PENALTY_BEFORE_INTERPOLATION ? density(penalty(1.0), xmin) : penalty(density(1.0, xmin)))
    end

    # Test with ProjectedPenalty
    @testset "ProjectedPenalty" begin
        base_penalty = PowerPenalty(3.0)
        proj = HeavisideProjection(5.0)
        penalty = ProjectedPenalty(base_penalty, proj)
        xmin = 0.001

        x_e = 0.5
        result = get_ρ(x_e, penalty, xmin)

        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected = density(penalty(x_e), xmin)
        else
            expected = penalty(density(x_e, xmin))
        end
        @test result ≈ expected

        # Test with different base penalties
        rp_base = RationalPenalty(2.0)
        rp_proj = ProjectedPenalty(rp_base, proj)
        result_rp = get_ρ(0.5, rp_proj, xmin)
        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            expected_rp = density(rp_proj(0.5), xmin)
        else
            expected_rp = rp_proj(density(0.5, xmin))
        end
        @test result_rp ≈ expected_rp
    end

    # Test type stability
    @testset "Type stability" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001
        x_e = 0.5

        result = get_ρ(x_e, penalty, xmin)
        @test result isa Float64

        # Test with Float32
        penalty_f32 = PowerPenalty(3.0f0)
        xmin_f32 = 0.001f0
        x_e_f32 = 0.5f0

        result_f32 = get_ρ(x_e_f32, penalty_f32, xmin_f32)
        @test result_f32 isa Float32
    end

    # Test mathematical properties
    @testset "Mathematical properties" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001

        # For any valid input, result should be between xmin and penalty(1.0) (or vice versa)
        for x_e in [0.0, 0.25, 0.5, 0.75, 1.0]
            result = get_ρ(x_e, penalty, xmin)
            # Result should be in reasonable range
            @test result >= min(xmin, penalty(xmin))
            @test result <= max(penalty(1.0), 1.0)
        end

        # Test monotonicity: higher x_e should give higher result (for typical penalties)
        # Note: This may not hold for all penalty/projection combinations
        x1, x2 = 0.3, 0.7
        r1 = get_ρ(x1, penalty, xmin)
        r2 = get_ρ(x2, penalty, xmin)
        @test r1 < r2  # PowerPenalty should preserve ordering
    end

    # Test consistency with manual computation
    @testset "Consistency with manual computation" begin
        penalty = PowerPenalty(2.0)
        xmin = 0.01
        x_e = 0.6

        ρ_computed = get_ρ(x_e, penalty, xmin)

        # Manual step-by-step computation
        if TopOpt.PENALTY_BEFORE_INTERPOLATION
            # First apply penalty, then density interpolation
            penalized = penalty(x_e)  # x_e^p
            ρ_manual = density(penalized, xmin)  # penalized * (1 - xmin) + xmin
        else
            # First apply density interpolation, then penalty
            interpolated = density(x_e, xmin)  # x_e * (1 - xmin) + xmin
            ρ_manual = penalty(interpolated)  # interpolated^p
        end

        @test ρ_computed ≈ ρ_manual
    end
end

@testset "get_ρ_dρ - penalized density with derivative" begin
    @info "Testing get_ρ_dρ with PENALTY_BEFORE_INTERPOLATION = $(TopOpt.PENALTY_BEFORE_INTERPOLATION)"

    @testset "PowerPenalty derivatives" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001

        # Test at x_e = 0.5
        x_e = 0.5
        ρ_val, dρ = get_ρ_dρ(x_e, penalty, xmin)

        # Verify the value matches get_ρ
        @test ρ_val ≈ get_ρ(x_e, penalty, xmin)

        # Verify derivative using finite differences
        ε = 1e-8
        ρ_plus = get_ρ(x_e + ε, penalty, xmin)
        ρ_minus = get_ρ(x_e - ε, penalty, xmin)
        fd_derivative = (ρ_plus - ρ_minus) / (2 * ε)

        @test dρ ≈ fd_derivative rtol = 1e-5

        # Test at other points
        for x_test in [0.1, 0.3, 0.7, 0.9]
            ρ_val_t, dρ_t = get_ρ_dρ(x_test, penalty, xmin)

            ρ_plus_t = get_ρ(x_test + ε, penalty, xmin)
            ρ_minus_t = get_ρ(x_test - ε, penalty, xmin)
            fd_derivative_t = (ρ_plus_t - ρ_minus_t) / (2 * ε)

            @test dρ_t ≈ fd_derivative_t rtol = 1e-5
        end
    end

    @testset "RationalPenalty derivatives" begin
        penalty = RationalPenalty(3.0)
        xmin = 0.001

        x_e = 0.5
        ρ_val, dρ = get_ρ_dρ(x_e, penalty, xmin)

        @test ρ_val ≈ get_ρ(x_e, penalty, xmin)

        ε = 1e-8
        ρ_plus = get_ρ(x_e + ε, penalty, xmin)
        ρ_minus = get_ρ(x_e - ε, penalty, xmin)
        fd_derivative = (ρ_plus - ρ_minus) / (2 * ε)

        @test dρ ≈ fd_derivative rtol = 1e-5
    end

    @testset "SinhPenalty derivatives" begin
        penalty = SinhPenalty(3.0)
        xmin = 0.001

        x_e = 0.5
        ρ_val, dρ = get_ρ_dρ(x_e, penalty, xmin)

        @test ρ_val ≈ get_ρ(x_e, penalty, xmin)

        ε = 1e-8
        ρ_plus = get_ρ(x_e + ε, penalty, xmin)
        ρ_minus = get_ρ(x_e - ε, penalty, xmin)
        fd_derivative = (ρ_plus - ρ_minus) / (2 * ε)

        @test dρ ≈ fd_derivative rtol = 1e-5
    end

    @testset "ProjectedPenalty derivatives" begin
        base_penalty = PowerPenalty(3.0)
        proj = HeavisideProjection(5.0)
        penalty = ProjectedPenalty(base_penalty, proj)
        xmin = 0.001

        x_e = 0.5
        ρ_val, dρ = get_ρ_dρ(x_e, penalty, xmin)

        @test ρ_val ≈ get_ρ(x_e, penalty, xmin)

        ε = 1e-8
        ρ_plus = get_ρ(x_e + ε, penalty, xmin)
        ρ_minus = get_ρ(x_e - ε, penalty, xmin)
        fd_derivative = (ρ_plus - ρ_minus) / (2 * ε)

        @test dρ ≈ fd_derivative rtol = 1e-5
    end

    @testset "Edge case derivatives" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001

        # Test at boundaries
        for x_e in [0.0, 1.0]
            ρ_val, dρ = get_ρ_dρ(x_e, penalty, xmin)
            @test ρ_val ≈ get_ρ(x_e, penalty, xmin)
            @test isfinite(dρ)
        end

        # Test with xmin = 0.0
        ρ_val, dρ = get_ρ_dρ(0.5, penalty, 0.0)
        @test ρ_val ≈ get_ρ(0.5, penalty, 0.0)
        @test isfinite(dρ)
    end
end

@testset "getprevpenalty" begin
    # Create a simple problem for testing
    nels = (2, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

    @testset "Initial state - prev_penalty equals penalty" begin
        # Test with DirectSolver
        solver = FEASolver(DirectSolver, problem)
        # Compare p values, not struct equality (they are different objects)
        @test getprevpenalty(solver).p == getpenalty(solver).p
        @test getprevpenalty(solver) isa PowerPenalty

        # Test with CGAssemblySolver
        solver_cg = FEASolver(CGAssemblySolver, problem)
        @test getprevpenalty(solver_cg).p == getpenalty(solver_cg).p

        # Test with CGMatrixFreeSolver
        solver_mf = FEASolver(CGMatrixFreeSolver, problem)
        @test getprevpenalty(solver_mf).p == getpenalty(solver_mf).p
    end

    @testset "After setpenalty! with number - prev_penalty stores old value" begin
        solver = FEASolver(DirectSolver, problem)
        
        # Get initial penalty
        initial_penalty = getpenalty(solver)
        initial_p = initial_penalty.p
        
        # Set new penalty value
        new_p = 3.0
        setpenalty!(solver, new_p)
        
        # Check that prev_penalty stores the old value
        @test getprevpenalty(solver).p == initial_p
        @test getpenalty(solver).p == new_p
        @test getprevpenalty(solver) != getpenalty(solver)
    end

    @testset "After setpenalty! with AbstractPenalty - prev_penalty stores old penalty object" begin
        solver = FEASolver(DirectSolver, problem)
        
        # Store reference to initial penalty
        old_penalty = getpenalty(solver)
        old_p = old_penalty.p
        
        # Create and set a new penalty object
        new_penalty = PowerPenalty(5.0)
        setpenalty!(solver, new_penalty)
        
        # prev_penalty should be a copy of the old penalty, not the same object
        prev = getprevpenalty(solver)
        @test prev.p == old_p
        @test getpenalty(solver) == new_penalty
        @test getpenalty(solver) !== old_penalty  # Different object
    end

    @testset "Multiple consecutive setpenalty! calls" begin
        solver = FEASolver(DirectSolver, problem)
        
        # First update
        setpenalty!(solver, 2.0)
        prev_after_1 = getprevpenalty(solver).p
        curr_after_1 = getpenalty(solver).p
        
        @test prev_after_1 == 1.0  # Default PowerPenalty has p=1
        @test curr_after_1 == 2.0
        
        # Second update
        setpenalty!(solver, 3.0)
        prev_after_2 = getprevpenalty(solver).p
        curr_after_2 = getpenalty(solver).p
        
        @test prev_after_2 == 2.0
        @test curr_after_2 == 3.0
    end

    @testset "Immutability - modifying current penalty doesn't affect previous" begin
        solver = FEASolver(DirectSolver, problem)
        
        # Set initial penalty
        setpenalty!(solver, 2.0)
        
        # Store reference to prev_penalty
        prev_before = getprevpenalty(solver)
        prev_p_before = prev_before.p
        
        # Modify current penalty directly
        getpenalty(solver).p = 99.0
        
        # prev_penalty should be unchanged
        @test getprevpenalty(solver).p == prev_p_before
    end

    @testset "Different penalty types" begin
        # Test with RationalPenalty
        rp = RationalPenalty(2.0)
        solver = FEASolver(DirectSolver, problem; penalty=rp)
        
        @test getprevpenalty(solver) isa RationalPenalty
        @test getprevpenalty(solver).p == 2.0
        
        setpenalty!(solver, 3.0)
        @test getprevpenalty(solver).p == 2.0
        @test getpenalty(solver).p == 3.0
        
        # Test with SinhPenalty
        sp = SinhPenalty(1.5)
        solver2 = FEASolver(DirectSolver, problem; penalty=sp)
        
        @test getprevpenalty(solver2) isa SinhPenalty
        setpenalty!(solver2, 2.5)
        @test getprevpenalty(solver2).p == 1.5
        @test getpenalty(solver2).p == 2.5
        
        # Test with ProjectedPenalty
        pp = ProjectedPenalty(PowerPenalty(2.0), HeavisideProjection(5.0))
        solver3 = FEASolver(DirectSolver, problem; penalty=pp)
        
        @test getprevpenalty(solver3) isa ProjectedPenalty
        @test getprevpenalty(solver3).p == 2.0
        
        setpenalty!(solver3, 4.0)
        @test getprevpenalty(solver3).p == 2.0
        @test getpenalty(solver3).p == 4.0
    end

    @testset "Custom prev_penalty initialization" begin
        # Create solver with custom prev_penalty
        custom_prev = PowerPenalty(5.0)
        solver = FEASolver(DirectSolver, problem; prev_penalty=custom_prev)
        
        @test getprevpenalty(solver).p == 5.0
        # After initialization with custom prev_penalty, it should be different from current
        @test getprevpenalty(solver) != getpenalty(solver)
    end

    @testset "Different physics types" begin
        # LinearElasticity (default from problem type)
        solver_struct = FEASolver(DirectSolver, problem)
        @test getprevpenalty(solver_struct).p == getpenalty(solver_struct).p
        
        # HeatTransfer (explicit physics type)
        solver_heat = FEASolver(HeatTransfer, DirectSolver, problem)
        @test getprevpenalty(solver_heat).p == getpenalty(solver_heat).p
        
        # Test setpenalty! with HeatTransfer
        setpenalty!(solver_heat, 2.0)
        @test getprevpenalty(solver_heat).p == 1.0
        @test getpenalty(solver_heat).p == 2.0
    end
end