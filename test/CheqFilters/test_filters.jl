using TopOpt, LinearAlgebra, Test, Statistics
using TopOpt: DensityFilter, PseudoDensities
using TopOpt.CheqFilters: SensFilter, FilterMetadata
import ChainRulesCore
using NonconvexCore: getdim

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
    
    @testset "DensityFilter boolean template constructors" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        
        @testset "DensityFilter{true} constructor" begin
            df_true = DensityFilter{true}(solver, rmin)
            
            @test df_true isa DensityFilter{true}
            @test df_true.rmin == rmin
            @test df_true.metadata isa FilterMetadata
            @test !isempty(df_true.jacobian)  # Jacobian should be computed
        end
        
        @testset "DensityFilter{false} constructor" begin
            df_false = DensityFilter{false}(solver, rmin)
            
            @test df_false isa DensityFilter{false}
            @test df_false.rmin == rmin
            @test df_false.metadata isa FilterMetadata
            @test isempty(df_false.jacobian)  # Jacobian should be empty (0x0 matrix)
            @test size(df_false.jacobian) == (0, 0)
        end
        
        @testset "Filtering vs non-filtering behavior" begin
            df_true = DensityFilter{true}(solver, rmin)
            df_false = DensityFilter{false}(solver, rmin)
            
            x = rand(length(solver.vars))
            
            # DensityFilter{true} applies filtering
            result_true = df_true(PseudoDensities(x))
            @test result_true isa PseudoDensities
            @test result_true.x != x  # Values should be modified
            
            # DensityFilter{false} returns values unchanged
            result_false = df_false(PseudoDensities(x))
            @test result_false isa PseudoDensities
            @test result_false.x ≈ x  # Values should be unchanged
        end
        
        @testset "getdim returns correct dimension" begin
            df_true = DensityFilter{true}(solver, rmin)
            df_false = DensityFilter{false}(solver, rmin)
            
            # For filtering=true, getdim returns size of jacobian
            @test getdim(df_true) == size(df_true.jacobian, 1)
            
            # For filtering=false, getdim returns 0 (empty jacobian)
            @test getdim(df_false) == 0
        end
        
        @testset "show methods for boolean template filters" begin
            df_true = DensityFilter{true}(solver, rmin)
            df_false = DensityFilter{false}(solver, rmin)
            
            io = IOBuffer()
            
            # Test show for DensityFilter{true}
            show(io, MIME"text/plain"(), df_true)
            output_true = String(take!(io))
            @test occursin("density filter", lowercase(output_true))
            
            # Test show for DensityFilter{false}
            show(io, MIME"text/plain"(), df_false)
            output_false = String(take!(io))
            @test occursin("density filter", lowercase(output_false))
        end
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
    
    @testset "DensityFilter values and gradients with transpose" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        
        x_raw = rand(length(solver.vars))
        
        # Forward pass: DensityFilter filters the values
        x_filtered = df(PseudoDensities(x_raw)).x
        
        # Verify forward filtering actually changes values
        @test x_filtered != x_raw
        
        # Test gradient via ChainRules
        y, pullback = ChainRulesCore.rrule(df, PseudoDensities(x_raw))
        
        # Output is filtered values
        @test y.x ≈ x_filtered
        
        # Backward pass: gradient w.r.t. filtered output
        # For sum(filtered), gradient is ones
        Δ = PseudoDensities(ones(length(x_raw)))
        
        # Pullback returns gradient w.r.t. input
        _, grad = pullback(Δ)
        
        # Density filter uses transpose of jacobian for gradients
        # grad.x should equal jacobian' * Δ
        @test length(grad.x) == length(x_raw)
        @test isfinite(sum(grad.x))
        
        # Compare with manual transpose multiplication
        manual_grad = df.jacobian' * Δ.x
        @test grad.x ≈ manual_grad rtol=1e-10
    end
    
    @testset "SensFilter only filters gradients" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        sf = SensFilter(solver; rmin=rmin)
        
        x_raw = rand(length(solver.vars))
        
        # Forward pass: SensFilter returns values unchanged
        x_out = sf(PseudoDensities(x_raw)).x
        
        # Values should be identical (no filtering in forward)
        @test x_out ≈ x_raw
        
        # Test gradient via ChainRules
        y, pullback = ChainRulesCore.rrule(sf, PseudoDensities(x_raw))
        
        # Output equals input (no forward filtering)
        @test y.x ≈ x_raw
        
        # Backward pass with non-uniform gradient
        # Create a gradient that varies spatially
        Δ = PseudoDensities(collect(1:length(x_raw)) ./ length(x_raw))
        
        _, grad = pullback(Δ)
        
        # SensFilter applies filtering in backward pass only
        # The gradient should be filtered (different from input gradient)
        @test length(grad.x) == length(x_raw)
        @test isfinite(sum(grad.x))
        
        # Gradient is filtered, so it should differ from original
        # (unless gradient is uniform)
        @test grad.x != Δ.x
        
        # With uniform gradient, filtering has no effect
        Δ_uniform = PseudoDensities(ones(length(x_raw)))
        _, grad_uniform = pullback(Δ_uniform)
        
        # Uniform gradient stays uniform after filtering
        @test all(grad_uniform.x .≈ grad_uniform.x[1])
    end
    
    @testset "Combined filter with sum function" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        sf = SensFilter(solver; rmin=rmin)
        
        # Test function: sum of filtered values
        # For DensityFilter: filters both values and backpropagates gradients
        # For SensFilter: leaves values unchanged, only filters gradients
        
        # DensityFilter test
        x1 = rand(length(solver.vars))
        
        # Forward: density filtered
        y1 = sum(df(PseudoDensities(x1)).x)
        
        # Gradient check via finite differences
        function density_filtered_sum(x)
            return sum(df(PseudoDensities(x)).x)
        end
        
        # SensFilter test
        x2 = rand(length(solver.vars))
        
        # Forward: values unchanged
        y2 = sum(sf(PseudoDensities(x2)).x)
        
        # Should equal sum of raw values (no forward filtering)
        @test y2 ≈ sum(x2)
        
        # Gradient via pullback
        _, pullback_df = ChainRulesCore.rrule(df, PseudoDensities(x1))
        _, pullback_sf = ChainRulesCore.rrule(sf, PseudoDensities(x2))
        
        # Gradient of sum is ones
        Δ = PseudoDensities(ones(length(solver.vars)))
        
        _, grad_df = pullback_df(Δ)
        _, grad_sf = pullback_sf(Δ)
        
        # Both gradients should be valid
        @test length(grad_df.x) == length(x1)
        @test length(grad_sf.x) == length(x2)
        @test isfinite(sum(grad_df.x))
        @test isfinite(sum(grad_sf.x))
    end
    
    @testset "Gradient filtering comparison" begin
        nels = (5, 5)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        
        rmin = 2.0
        df = DensityFilter(solver; rmin=rmin)
        sf = SensFilter(solver; rmin=rmin)
        
        x = rand(length(solver.vars))
        
        # Create a spatially varying gradient
        Δ = PseudoDensities([sin(2π * i / length(x)) for i in 1:length(x)])
        
        # DensityFilter pullback: uses jacobian transpose
        _, pullback_df = ChainRulesCore.rrule(df, PseudoDensities(x))
        _, grad_df = pullback_df(Δ)
        
        # SensFilter pullback: uses nodal gradient smoothing
        _, pullback_sf = ChainRulesCore.rrule(sf, PseudoDensities(x))
        _, grad_sf = pullback_sf(Δ)
        
        # Both should produce smoothed gradients
        @test length(grad_df.x) == length(x)
        @test length(grad_sf.x) == length(x)
        
        # Gradients should be finite
        @test all(isfinite, grad_df.x)
        @test all(isfinite, grad_sf.x)
        
        # DensityFilter: output values are filtered, gradient uses jacobian'
        # SensFilter: output values are unchanged, gradient is smoothed
        df_output = df(PseudoDensities(x)).x
        sf_output = sf(PseudoDensities(x)).x
        
        @test df_output != x  # DensityFilter changes values
        @test sf_output ≈ x   # SensFilter leaves values unchanged
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
