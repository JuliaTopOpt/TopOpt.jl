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