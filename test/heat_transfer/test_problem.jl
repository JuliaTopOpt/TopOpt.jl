using Test, TopOpt, Ferrite, LinearAlgebra

@testset "Heat Conduction Problem Setup" begin
    # Create a simple 2D heat conduction problem
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    heat_source = 1.0

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k, heat_source;
        Tleft=0.0, Tright=0.0
    )

    @test problem isa HeatConductionProblem
    @test getk(problem) ≈ 1.0
    @test getheat_source(problem) ≈ 1.0
    @test Ferrite.getncells(problem) == 50
end

@testset "Heat Transfer Element Matrices" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    k = 1.0
    heat_source = 1.0

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k, heat_source;
        Tleft=0.0, Tright=0.0
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
    heat_source = 1.0

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k, heat_source;
        Tleft=0.0, Tright=0.0
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

    # With heat generation and zero boundary conditions,
    # temperature should be highest in the center
    @test maximum(solver.u) > 0.0
end

@testset "Thermal Compliance Function" begin
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    heat_source = 1.0

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k, heat_source;
        Tleft=0.0, Tright=0.0
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

@testset "Heat Conduction Analytical Solution" begin
    # 1D heat conduction with uniform heat generation
    # T(x) = q*x*(L-x)/(2*k) for fixed temperature BCs
    # Testing 2D problem that's effectively 1D

    nels = (20, 1)  # Thin strip
    sizes = (1.0, 0.1)  # Length 1, small height
    k = 1.0
    q = 1.0
    L = nels[1] * sizes[1]

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k, q;
        Tleft=0.0, Tright=0.0
    )

    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    solver.vars .= 1.0
    solver()

    # Get maximum temperature at center
    T_max_numerical = maximum(solver.u)

    # Analytical maximum at x = L/2: T_max = q*L^2/8
    T_max_analytical = q * L^2 / 8

    # Allow some error due to discretization
    @test isapprox(T_max_numerical, T_max_analytical; rtol=0.1)
end
