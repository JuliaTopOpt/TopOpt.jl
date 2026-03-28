using TopOpt, Test

@testset "CGAssemblySolver with safe=true" begin
    @testset "Basic structural problem with safe mode" begin
        nels = (20, 10, 10)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Create CGAssemblySolver
        solver = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        
        # Set random density variables
        x0 = rand(length(solver.vars))
        solver.vars .= x0
        
        # Call solver with safe=true using Val{true}
        solver(false, Val{true})
        
        # Verify solution is not NaN or Inf
        @test all(isfinite, solver.u)
        @test !isempty(solver.u)
    end

    @testset "Compare safe=true vs safe=false" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Create two identical solvers
        solver_safe = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        solver_normal = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        
        # Set same density variables
        x0 = fill(0.5, length(solver_safe.vars))
        solver_safe.vars .= x0
        solver_normal.vars .= x0
        
        # Solve with safe=true
        solver_safe(false, Val{true})
        
        # Solve with safe=false (default)
        solver_normal(false, Val{false})
        
        # Both should produce finite results
        @test all(isfinite, solver_safe.u)
        @test all(isfinite, solver_normal.u)
        
        # Results should be similar for well-conditioned problems
        @test solver_safe.u ≈ solver_normal.u rtol=1e-4
    end

    @testset "Heat Transfer problem with safe mode" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        solver = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        solver.vars .= 1.0
        
        # Call with safe=true
        solver(false, Val{true})
        
        # Verify solution
        @test all(isfinite, solver.u)
        @test !isempty(solver.u)
    end

    @testset "Low density values with safe mode" begin
        # Test with very low densities that might cause numerical issues
        nels = (8, 4, 4)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        solver = FEASolver(CGAssemblySolver, problem; abstol=1e-7, xmin=1e-6)
        
        # Set very low densities
        solver.vars .= 1e-5
        
        # This should still work with safe=true
        solver(false, Val{true})
        
        @test all(isfinite, solver.u)
    end
end

println("CGAssemblySolver safe=true tests completed successfully!")