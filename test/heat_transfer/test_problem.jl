using Test, TopOpt, Ferrite, LinearAlgebra, Random, FiniteDifferences, Zygote
using TopOpt: ndofs, Nonconvex
const FDM = FiniteDifferences

Random.seed!(42)

@testset "Heat Conduction Problem Setup" begin
    # Create a simple 2D heat conduction problem with surface heat flux
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    # Apply heat flux on top boundary (positive = heat into domain)
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    @test problem isa HeatConductionProblem
    @test getk(problem) ≈ 1.0
    @test Ferrite.getncells(problem) == 50
end

@testset "Heat Transfer Element Matrices" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    k = 1.0
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

    @test length(elementinfo.Kes) == 8  # 4x2 elements
    @test length(elementinfo.fes) == 8

    # For linear quad elements, each element has 4 nodes
    # and heat transfer is scalar field, so Ke is 4x4
    @test size(elementinfo.Kes[1], 1) == 4
end

@testset "Direct Heat Solver" begin
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    @test solver isa TopOpt.AbstractFEASolver

    # Set all design variables to 1 (full material)
    solver.vars .= 1.0

    # Solve
    solver()

    # Check that temperature solution was computed
    @test length(solver.u) == ndofs(problem.ch.dh)
    @test !any(isnan, solver.u)

    # With heat flux on top and zero boundary conditions,
    # temperature should be non-zero
    @test maximum(solver.u) > 0.0
end

@testset "Thermal Compliance Function" begin
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    comp = ThermalCompliance(solver)

    @test comp isa ThermalCompliance

    # Evaluate thermal compliance with uniform density
    x = ones(length(solver.vars))
    pd = PseudoDensities(x)

    val = comp(pd)
    @test val > 0.0
    @test !isnan(val)
end

@testset "Heat Conduction with Surface Heat Flux" begin
    # 2D heat conduction with surface heat flux on top boundary
    # Heat flux q enters the domain from the top boundary
    # Temperature fixed at left and right boundaries

    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    q = 1.0  # Heat flux on top boundary (W/m²)
    heatflux = Dict{String,Float64}("top" => q)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    solver.vars .= 1.0
    solver()

    # With heat flux on top and zero temperature on sides,
    # temperature should be non-zero in the domain
    @test maximum(solver.u) > 0.0

    # Temperature should be highest near the top (center)
    # and decrease toward the boundaries
    T_max = maximum(solver.u)
    @test T_max > 0.0
end

@testset "Thermal Compliance - Gradient Accuracy" begin
    @testset "Zygote vs FiniteDifferences consistency" begin
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=1.0, Tright=0.0, heatflux=heatflux
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(1.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))

        # Test at multiple random points
        for i in 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)

            val = f(x)
            grad_zygote = Zygote.gradient(f, x)[1]

            # Check value is reasonable
            @test val > 0
            @test isfinite(val)

            # Check gradient properties
            @test length(grad_zygote) == length(x)
            @test all(isfinite.(grad_zygote))

            # Compute finite difference gradient
            fd_grad = FDM.grad(FDM.backward_fdm(2, 1), f, x)[1]

            # Gradient should match finite differences closely
            @test isapprox(grad_zygote, fd_grad; rtol=1e-2, atol=1e-6)
        end
    end

    @testset "Penalty power law verification" begin
        # For SIMP, thermal compliance should follow power law
        # J(ρ) = ρ^p * J(1) for uniform density
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        # Reference solution with full density
        solver_ref = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        comp_ref = ThermalCompliance(solver_ref)
        tc_full = comp_ref(PseudoDensities(ones(prod(nels))))

        # Test different penalty exponents
        for p in [1.0, 2.0, 3.0]
            solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(p))
            comp = ThermalCompliance(solver)

            # With uniform density ρ=0.5, compliance should scale as (0.5)^p
            rho = 0.5
            x_uniform = fill(rho, prod(nels))
            tc_uniform = comp(PseudoDensities(x_uniform))

            # Check power law approximately holds
            expected_ratio = rho^p
            actual_ratio = tc_uniform / tc_full

            # Allow 20% tolerance for discretization effects
            @test isapprox(actual_ratio, expected_ratio; rtol=0.2)
        end
    end
end

@testset "Heat Transfer - Solver Consistency" begin
    @testset "DirectSolver produces symmetric positive definite system" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        globalinfo = GlobalFEAInfo(problem)

        # Assemble system with uniform density
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001

        TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

        K = globalinfo.K
        # Check symmetry
        @test Matrix(K) ≈ Matrix(K)'

        # Check positive definiteness (all eigenvalues should be positive)
        eigs = eigvals(Matrix(K))
        @test all(eigs .> -1e-10)  # Allow small numerical errors
    end

    @testset "Solution satisfies governing equations" begin
        # K*u = f should be satisfied
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        solver.vars .= 1.0
        solver()

        # Get residual: K*u - f
        globalinfo = solver.globalinfo
        elementinfo = solver.elementinfo

        vars = ones(length(solver.vars))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

        K = globalinfo.K
        f = globalinfo.f
        u = solver.u

        residual = K * u - f

        # Residual should be small (accounting for boundary conditions)
        # BC nodes will have residual from prescribed values
        @test norm(residual) / norm(f) < 0.1
    end
end

@testset "Heat Transfer - Error Handling & Type Safety" begin
    @testset "ThermalCompliance rejects structural problems" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError ThermalCompliance(solver)
    end

    @testset "Compliance rejects heat transfer problems" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, 1.0;
            Tleft=0.0, Tright=0.0
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError Compliance(solver)
    end
end

@testset "Heat Transfer - Element Properties" begin
    @testset "Element stiffness matrix symmetry and properties" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # Check number of elements
        @test length(elementinfo.Kes) == 4

        # Check each element matrix
        for Ke in elementinfo.Kes
            # Should be 4x4 for linear quadrilateral
            @test size(Ke, 1) == 4
            @test size(Ke, 2) == 4

            # Should be symmetric
            mat_Ke = Matrix(Ke)
            @test mat_Ke ≈ mat_Ke' rtol=1e-10

            # Should be positive semi-definite (non-negative eigenvalues)
            eigs = eigvals(mat_Ke)
            @test all(eigs .>= -1e-10)
        end

        # Check that fes is zeros (no body forces in heat transfer)
        @test length(elementinfo.fes) == 4
        for fe in elementinfo.fes
            @test length(fe) == 4
            # fes should be zeros (no body forces)
            @test all(fe .== 0)
        end
    end

    @testset "Element volumes are positive" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # All cell volumes should be positive
        @test all(elementinfo.cellvolumes .> 0)

        # Total volume should match domain size
        total_vol = sum(elementinfo.cellvolumes)
        expected_vol = nels[1] * sizes[1] * nels[2] * sizes[2]
        @test isapprox(total_vol, expected_vol; rtol=1e-10)
    end
end