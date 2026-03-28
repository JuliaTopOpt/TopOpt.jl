using TopOpt, Test, LinearAlgebra
using TopOpt.FEA: EnergyCriteria, CGAssemblySolver, CGMatrixFreeSolver, DirectSolver

@testset "CG with EnergyCriteria" begin
    @testset "CGAssemblySolver with EnergyCriteria" begin
        nels = (20, 10, 10)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Create solver with EnergyCriteria
        solver_energy = FEASolver(CGAssemblySolver, problem; 
            abstol=1e-7, 
            conv=EnergyCriteria()
        )
        
        # Create reference solver with DefaultCriteria
        solver_default = FEASolver(CGAssemblySolver, problem; 
            abstol=1e-7
        )
        
        # Create direct solver for reference solution
        solver_direct = FEASolver(DirectSolver, problem)

        x0 = rand(length(solver_energy.vars))
        solver_energy.vars .= x0
        solver_default.vars .= x0
        solver_direct.vars .= x0

        solver_energy()
        solver_default()
        solver_direct()

        # Test that EnergyCriteria produces valid solution
        @test !any(isnan, solver_energy.u)
        @test !any(isinf, solver_energy.u)
        
        # Test that solutions are approximately equal
        @test solver_energy.u ≈ solver_default.u rtol=1e-4
        @test solver_energy.u ≈ solver_direct.u rtol=1e-4
    end

    @testset "CGMatrixFreeSolver with EnergyCriteria" begin
        nels = (20, 10, 10)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Create solver with EnergyCriteria
        solver_energy = FEASolver(CGMatrixFreeSolver, problem; 
            abstol=1e-7, 
            conv=EnergyCriteria()
        )
        
        # Create reference solver with DefaultCriteria
        solver_default = FEASolver(CGMatrixFreeSolver, problem; 
            abstol=1e-7
        )
        
        # Create direct solver for reference solution
        solver_direct = FEASolver(DirectSolver, problem)

        x0 = rand(length(solver_energy.vars))
        solver_energy.vars .= x0
        solver_default.vars .= x0
        solver_direct.vars .= x0

        solver_energy()
        solver_default()
        solver_direct()

        # Test that EnergyCriteria produces valid solution
        @test !any(isnan, solver_energy.u)
        @test !any(isinf, solver_energy.u)
        
        # Test that solutions are approximately equal
        @test solver_energy.u ≈ solver_default.u rtol=1e-4
        @test solver_energy.u ≈ solver_direct.u rtol=1e-4
    end

    @testset "EnergyCriteria energy tracking" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Create solver with EnergyCriteria
        energy_criteria = EnergyCriteria()
        solver = FEASolver(CGAssemblySolver, problem; 
            abstol=1e-7, 
            conv=energy_criteria
        )

        x0 = rand(length(solver.vars))
        solver.vars .= x0

        # Energy should be 0.0 initially
        @test energy_criteria.energy == 0.0
        
        # Run solver - energy will be updated during CG iterations
        solver()
        
        # After solving, the energy criteria should have been updated
        # Note: The actual value depends on the problem, but it should be non-negative
        # since energy is computed as xAx which is positive definite
        @test energy_criteria.energy >= 0
    end

    @testset "EnergyCriteria convergence behavior" begin
        nels = (6, 4)
        sizes = (1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        
        # Test with different tolerances
        for abstol in [1e-5, 1e-7, 1e-9]
            solver_energy = FEASolver(CGAssemblySolver, problem; 
                abstol=abstol, 
                conv=EnergyCriteria()
            )
            solver_direct = FEASolver(DirectSolver, problem)
            
            x0 = rand(length(solver_energy.vars))
            solver_energy.vars .= x0
            solver_direct.vars .= x0
            
            solver_energy()
            solver_direct()
            
            # Tighter tolerance should give closer solution to direct
            @test solver_energy.u ≈ solver_direct.u rtol=sqrt(abstol)
        end
    end
end