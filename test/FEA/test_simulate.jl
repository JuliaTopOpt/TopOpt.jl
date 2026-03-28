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
end
