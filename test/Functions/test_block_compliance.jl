using TopOpt, Test, Random, SparseArrays
using LinearAlgebra
import Nonconvex

Random.seed!(42)

@testset "BlockCompliance API Tests" begin
    @testset "Single load case - basic functionality" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create multi-load problem with single load
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        @test bc isa BlockCompliance
        @test Nonconvex.NonconvexCore.getdim(bc) == 1
        
        # Test evaluation with uniform density
        x = ones(prod(nels))
        result = bc(PseudoDensities(x))
        @test length(result) == 1
        @test isfinite(result[1])
        @test result[1] > 0
    end

    @testset "getpenalty" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        current_penalty = TopOpt.Utilities.getpenalty(bc)
        @test current_penalty isa PowerPenalty
        @test current_penalty.p == 3.0
    end

    @testset "Different material distributions" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        # Test with uniform density
        x_uniform = ones(prod(nels))
        result_uniform = bc(PseudoDensities(x_uniform))
        @test isfinite(result_uniform[1])
        
        # Test with varying density
        x_varying = fill(0.5, prod(nels))
        result_varying = bc(PseudoDensities(x_varying))
        @test isfinite(result_varying[1])
    end
end

@testset "BlockCompliance Physical Properties" begin
    @testset "Different densities produce finite results" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        # Test different densities produce finite, positive results
        x_low = fill(0.3, prod(nels))
        x_high = fill(0.8, prod(nels))
        
        result_low = bc(PseudoDensities(x_low))
        result_high = bc(PseudoDensities(x_high))
        
        @test isfinite(result_low[1])
        @test isfinite(result_high[1])
        @test result_low[1] > 0
        @test result_high[1] > 0
    end
end

@testset "BlockCompliance with different problem types" begin
    @testset "Half MBB problem - single load" begin
        nels = (4, 4)
        base_problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01)
        bc = BlockCompliance(problem, solver; nv=1)
        
        x = ones(prod(nels))
        result = bc(PseudoDensities(x))
        @test length(result) == 1
        @test isfinite(result[1])
        @test result[1] > 0
    end
end