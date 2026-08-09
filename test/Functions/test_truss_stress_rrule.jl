using TopOpt, Zygote, FiniteDifferences, LinearAlgebra, Test, Random
const FDM = FiniteDifferences

Random.seed!(42)

@testset "TrussStress rrule" begin
    # Use the same simple 3-element truss as test_truss_stress_fns.jl
    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        joinpath(@__DIR__, "testfile2_compact.json")
    )
    loads = load_cases["0"]
    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
    )
    ncells = length(elements)

    xmin = 0.0001
    solver = FEASolver(DirectSolver, problem; xmin=xmin, penalty=PowerPenalty(1.0))
    ts = TrussStress(solver)

    # 1. Forward pass: stress with all-solid design matches hand calculation
    x_ones = ones(ncells)
    σ = ts(PseudoDensities(x_ones))
    expected = [-50 * sqrt(2.0), 50.0, -50 * sqrt(2.0)]
    @test σ ≈ expected atol = 1e-10

    # 2. rrule returns correct forward value
    σ_rr, pullback = ChainRulesCore.rrule(ts, PseudoDensities(x_ones))
    @test σ_rr ≈ σ atol = 1e-10

    # 3. Gradient correctness via finite differences
    for _ in 1:3
        x = clamp.(rand(ncells), 0.2, 1.0)
        # Scalar objective: weighted sum of stresses
        weights = randn(ncells)
        f = x_vec -> sum(ts(PseudoDensities(x_vec)) .* weights)

        val = f(x)
        @test isfinite(val)

        # Zygote gradient (uses rrule)
        grad_zygote = Zygote.gradient(f, x)[1]
        @test all(isfinite, grad_zygote)

        # Finite-difference gradient
        grad_fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]
        @test grad_zygote ≈ grad_fd rtol = 1e-4
    end

    # 4. Stress scales correctly with density (penalty p=1)
    x_half = fill(0.5, ncells)
    σ_half = ts(PseudoDensities(x_half))
    penalty = TopOpt.getpenalty(solver)
    ρ_half = TopOpt.Utilities.density(penalty(0.5), xmin)
    # With lower density, the stiffness is lower, so displacements are larger
    # and stress changes. Just check the result is finite and different.
    @test all(isfinite, σ_half)
    @test σ_half != σ
end

@testset "TrussStress rrule with higher penalty" begin
    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        joinpath(@__DIR__, "testfile2_compact.json")
    )
    loads = load_cases["0"]
    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
    )
    ncells = length(elements)

    # Use p=3 penalty to test that the rrule correctly chains through penalty
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
    ts = TrussStress(solver)

    for _ in 1:3
        x = clamp.(rand(ncells), 0.3, 1.0)
        weights = randn(ncells)
        f = x_vec -> sum(ts(PseudoDensities(x_vec)) .* weights)

        val = f(x)
        @test isfinite(val)

        grad_zygote = Zygote.gradient(f, x)[1]
        @test all(isfinite, grad_zygote)

        grad_fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]
        @test grad_zygote ≈ grad_fd rtol = 1e-3
    end
end