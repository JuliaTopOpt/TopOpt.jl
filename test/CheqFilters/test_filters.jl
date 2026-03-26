using TopOpt, LinearAlgebra, Test, Statistics
using TopOpt: DensityFilter, PseudoDensities
using TopOpt.CheqFilters: SensFilter, FilterMetadata

@testset "Filter Tests" begin
    @testset "DensityFilter Construction" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        
        @test df.rmin == rmin
        @test df.metadata isa FilterMetadata
    end
    
    @testset "SensFilter Construction" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        sf = SensFilter(solver; rmin=rmin)
        
        @test sf.rmin == rmin
        @test sf.metadata isa FilterMetadata
    end
    
    @testset "DensityFilter Application" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        
        n = length(solver.vars)
        x = ones(n) * 0.5
        
        result = df(PseudoDensities(x))
        
        @test result isa PseudoDensities
        @test length(result.x) == n
        
        @test all(result.x .>= 0.0)
        @test all(result.x .<= 1.0)
    end
    
    @testset "SensFilter Application" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        sf = SensFilter(solver; rmin=rmin)
        
        n = length(solver.vars)
        x = ones(n) * 0.5
        
        result = sf(PseudoDensities(x))
        
        @test result isa PseudoDensities
        @test length(result.x) == n
    end
    
    @testset "Filter radius effects" begin
        nels = (10, 10)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        df_small = DensityFilter(solver; rmin=1.5)
        df_large = DensityFilter(solver; rmin=5.0)
        
        x = rand(length(solver.vars))
        
        result_small = df_small(PseudoDensities(x))
        result_large = df_large(PseudoDensities(x))
        
        @test result_small isa PseudoDensities
        @test result_large isa PseudoDensities
        
        std_small = std(result_small.x)
        std_large = std(result_large.x)
        @test std_large <= std_small * 1.5
    end
    
    @testset "Filter with uniform density" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        
        n = length(solver.vars)
        x_uniform = ones(n) * 0.5
        
        result = df(PseudoDensities(x_uniform))
        
        @test all(result.x .≈ 0.5)
    end
    
    @testset "Filter gradient check" begin
        nels = (3, 3)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        
        x = rand(length(solver.vars))
        
        f = x -> sum(df(PseudoDensities(x)).x)
        
        result = f(x)
        @test isfinite(result)
    end
    
    @testset "Multiple filters on same grid" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        df1 = DensityFilter(solver; rmin=1.5)
        df2 = DensityFilter(solver; rmin=3.0)
        
        x = rand(length(solver.vars))
        
        result1 = df1(PseudoDensities(x))
        result2 = df2(PseudoDensities(x))
        
        @test result1 isa PseudoDensities
        @test result2 isa PseudoDensities
        
        @test result1.x != result2.x
    end
end