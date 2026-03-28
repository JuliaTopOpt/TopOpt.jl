using Test, TopOpt, LinearAlgebra
using TopOpt.FEA: simulate

@testset "simulate function - basic tests" begin
    @testset "DirectSolver with PointLoadCantilever" begin
        nels = (6, 4)  # Even numbers for both dimensions
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        topology = ones(prod(nels))
        
        result = simulate(problem, topology)
        
        @test result isa TopOpt.FEA.LinearElasticityResult
        @test result.comp > 0
        @test length(result.u) > 0
    end

    @testset "Different topologies produce different results" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        full_topology = ones(prod(nels))
        zero_topology = zeros(prod(nels))
        
        # Full vs zero should give different displacement magnitudes
        result_full = simulate(problem, full_topology)
        result_void_hard_safe = simulate(problem, zero_topology, safe = true)
        result_void_hard_unsafe = simulate(problem, zero_topology, safe = false)
        result_void_soft = simulate(problem, zero_topology, hard = false)
        
        @test norm(result_full.u) > 0
        # Soft void has large displacement
        @test norm(result_void_soft.u) > norm(result_full.u) * 3
        # Hard void has NaN displacement
        @test all(isnan, result_void_hard_safe.u)
        @test all(isnan, result_void_hard_unsafe.u)
    end

    @testset "LinearElasticityResult properties" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        topology = ones(prod(nels))
        
        result = simulate(problem, topology)
        
        # Test that show method works without errors
        io = IOBuffer()
        retval = show(io, MIME"text/plain"(), result)
        @test retval === nothing  # show returns nothing
        
        # Test result has expected fields
        @test hasfield(typeof(result), :u)
        @test hasfield(typeof(result), :comp)
        @test result.comp isa Real
        @test result.u isa AbstractVector{<:Real}
    end

    @testset "round=false preserves continuous densities" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create topology with intermediate densities
        ncells = prod(nels)
        topology_continuous = fill(0.5, ncells)
        
        # With round=false, intermediate densities should be preserved
        result_continuous = simulate(problem, topology_continuous, round=false)
        
        @test result_continuous isa TopOpt.FEA.LinearElasticityResult
        @test result_continuous.comp > 0
        @test length(result_continuous.u) > 0
        
        # Compare with rounded version (0.5 rounds to 0, giving soft material)
        result_rounded = simulate(problem, topology_continuous, round=true, hard=false)
        
        # The compliance values should be different due to different material distributions
        @test result_continuous.comp != result_rounded.comp
    end

    @testset "round=false vs round=true with mixed topology" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        ncells = prod(nels)
        # Mix of solid (1.0), void (0.0), and intermediate densities
        topology_mixed = vcat(
            fill(1.0, div(ncells, 4)),      # solid
            fill(0.3, div(ncells, 4)),      # intermediate
            fill(0.7, div(ncells, 4)),      # intermediate
            fill(0.0, ncells - 3 * div(ncells, 4))  # void
        )
        
        # Round=false: preserves 0.3 and 0.7
        result_no_round = simulate(problem, topology_mixed, round=false)
        
        # Round=true: 0.3 -> 0, 0.7 -> 1
        result_round = simulate(problem, topology_mixed, round=true)
        
        @test result_no_round isa TopOpt.FEA.LinearElasticityResult
        @test result_round isa TopOpt.FEA.LinearElasticityResult
        
        # Results should differ because material distribution is different
        @test result_no_round.comp != result_round.comp
        @test norm(result_no_round.u) != norm(result_round.u)
    end

    @testset "round=false with uniform intermediate density" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        ncells = prod(nels)
        
        # Test various intermediate densities
        for density in [0.25, 0.5, 0.75]
            topology = fill(density, ncells)
            result = simulate(problem, topology, round=false)
            
            @test result isa TopOpt.FEA.LinearElasticityResult
            @test result.comp > 0
            @test all(isfinite, result.u)
            
            # Higher density should give lower compliance (stiffer structure)
            if density > 0.25
                prev_result = simulate(problem, fill(0.25, ncells), round=false)
                @test result.comp < prev_result.comp
            end
        end
    end
end
