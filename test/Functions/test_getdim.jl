using TopOpt, Nonconvex, Test
using Nonconvex: NonconvexCore, getdim

# Minimal test for Nonconvex.NonconvexCore.getdim(::Compliance) = 1
@testset "getdim Compliance test" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    comp = Compliance(solver)
    
    # Test Nonconvex.NonconvexCore.getdim(::Compliance) = 1
    dim = Nonconvex.NonconvexCore.getdim(comp)
    @test dim == 1
    println("✓ Nonconvex.NonconvexCore.getdim(::Compliance) = $dim")
    
    # Also test that getdim returns 1 for Volume
    vol = Volume(solver)
    vol_dim = getdim(vol)
    @test vol_dim == 1
    println("✓ getdim(::Volume) = $vol_dim")
end

# Test for Nonconvex.NonconvexCore.getdim(::MeanCompliance) = 1
@testset "getdim MeanCompliance test" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    multiload_problem = MultiLoad(problem, F -> begin
        return [F[:, 1] F[:, 1]]
    end)
    solver = FEASolver(DirectSolver, multiload_problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    mean_comp = MeanCompliance(multiload_problem, solver; method=:exact)
    
    # Test Nonconvex.NonconvexCore.getdim(::MeanCompliance) = 1
    dim = Nonconvex.NonconvexCore.getdim(mean_comp)
    @test dim == 1
    println("✓ Nonconvex.NonconvexCore.getdim(::MeanCompliance) = $dim")
end
