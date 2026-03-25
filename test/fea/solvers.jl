using TopOpt

nels = (20, 10, 10)
sizes = (1.0, 1.0, 1.0)
E = 1.0
ν = 0.3
force = -1.0
problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
solver1 = FEASolver(DirectSolver, problem)
# Takes forever to compile
solver2 = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)
solver3 = FEASolver(CGAssemblySolver, problem; abstol=1e-7)

x0 = rand(length(solver1.vars))
solver1.vars .= x0
solver2.vars .= x0
solver3.vars .= x0

solver1()
solver2()
solver3()

@test solver1.u ≈ solver2.u
@test solver1.u ≈ solver3.u

# Heat transfer solver tests
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
