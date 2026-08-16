using TopOpt, Test, LinearAlgebra, Random, FiniteDifferences, Zygote
using TopOpt: ndofs, PseudoDensities
using TopOpt.TopOptProblems: getdh, getncells

const FDM = FiniteDifferences

Random.seed!(42)

# Helper: compare Zygote gradient to 5th-order central finite differences.
function grad_vs_fd(name, f, x; rtol=1e-6, atol=1e-8)
    val = f(x)
    gz = Zygote.gradient(f, x)[1]
    fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]
    @test isfinite(val)
    @test all(isfinite, gz)
    @test length(gz) == length(x)
    @test isapprox(gz, fd; rtol=rtol, atol=atol)
    return val, gz
end

@testset "TemperatureFun" begin
    @testset "Construction validates problem type" begin
        # Structural problems are rejected.
        structprob = PointLoadCantilever((4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
        structsolver = FEASolver(DirectSolver, structprob)
        @test_throws ArgumentError TemperatureFun(structsolver)
    end

    @testset "Gradient matches finite differences: homogeneous Dirichlet" begin
        nels = (8, 6)
        problem = HeatConductionProblem(
            nels, (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 100.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(1.0))
        tf = TemperatureFun(solver)
        f = x -> sum(tf(PseudoDensities(x)))
        for _ in 1:3
            x = clamp.(rand(prod(nels)), 0.2, 1.0)
            grad_vs_fd("homog", f, x)
        end
    end

    @testset "Gradient matches finite differences: inhomogeneous Dirichlet" begin
        nels = (8, 6)
        problem = HeatConductionProblem(
            nels, (1.0, 1.0), 1.0; Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
        tf = TemperatureFun(solver)
        f = x -> sum(tf(PseudoDensities(x)))
        for _ in 1:3
            x = clamp.(rand(prod(nels)), 0.2, 1.0)
            grad_vs_fd("inhomog", f, x; rtol=1e-5)
        end
    end

    @testset "Returns nodal temperature of correct shape" begin
        nels = (6, 4)
        problem = HeatConductionProblem(
            nels, (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 100.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01)
        tf = TemperatureFun(solver)
        T = tf(PseudoDensities(ones(prod(nels))))
        @test T isa TopOpt.Functions.TemperatureResult
        @test length(T) == ndofs(problem.ch.dh)
        @test all(isfinite, T.T)
    end

    @testset "cell_temperature averages to per-cell vector" begin
        nels = (6, 4)
        problem = HeatConductionProblem(
            nels, (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 100.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01)
        tf = TemperatureFun(solver)
        T = tf(PseudoDensities(ones(prod(nels))))
        cellT = TopOpt.cell_temperature(T, problem)
        @test length(cellT) == getncells(problem)
        @test all(isfinite, cellT)
        # Every cell temperature lies between the min and max nodal temperature.
        Tmin, Tmax = minimum(T.T), maximum(T.T)
        @test all(Tmin <= c <= Tmax for c in cellT)
    end
end
