using TopOpt, Test, LinearAlgebra
using Ferrite: ndofs

# Access Preconditioners through TopOpt.FEA (it's imported in src/FEA/FEA.jl)
const Preconditioners = TopOpt.FEA.Preconditioners

@testset "CG Solvers with Preconditioner" begin
    @testset "CGAssemblySolver with diagonal preconditioner" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create a diagonal preconditioner (all ones on diagonal - acts like identity)
        # but is NOT === identity
        dummy_prec = Preconditioners.DiagonalPreconditioner(ones(Float64, ndofs(problem.ch.dh)))

        # Verify the preconditioner is not === identity
        @test !(dummy_prec === identity)

        # Create solver with preconditioner
        solver_prec = FEASolver(CGAssemblySolver, problem; abstol=1e-7, preconditioner=dummy_prec)
        
        # Create solver with identity preconditioner for comparison
        solver_identity = FEASolver(CGAssemblySolver, problem; abstol=1e-7, preconditioner=identity)

        # Set same density variables
        x0 = fill(0.5, length(solver_prec.vars))
        solver_prec.vars .= x0
        solver_identity.vars .= x0

        # Solve with preconditioner
        solver_prec()

        # Solve with identity
        solver_identity()

        # Both should produce finite results
        @test all(isfinite, solver_prec.u)
        @test all(isfinite, solver_identity.u)

        # Results should be approximately equal (since dummy_prec ≈ identity)
        @test solver_prec.u ≈ solver_identity.u rtol=1e-5
        
        # Verify preconditioner was initialized
        @test solver_prec.preconditioner_initialized[] == true
    end

    @testset "CGMatrixFreeSolver with diagonal preconditioner" begin
        nels = (8, 4, 4)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create a diagonal preconditioner (all ones on diagonal - acts like identity)
        # but is NOT === identity
        dummy_prec = Preconditioners.DiagonalPreconditioner(ones(Float64, ndofs(problem.ch.dh)))

        # Verify the preconditioner is not === identity
        @test !(dummy_prec === identity)

        # Create solver with preconditioner
        solver_prec = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7, preconditioner=dummy_prec)
        
        # Create solver with identity preconditioner for comparison
        solver_identity = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7, preconditioner=identity)

        # Set same density variables
        x0 = fill(0.5, length(solver_prec.vars))
        solver_prec.vars .= x0
        solver_identity.vars .= x0

        # Solve with preconditioner
        solver_prec()

        # Solve with identity
        solver_identity()

        # Both should produce finite results
        @test all(isfinite, solver_prec.u)
        @test all(isfinite, solver_identity.u)

        # Results should be approximately equal (since dummy_prec ≈ identity)
        @test solver_prec.u ≈ solver_identity.u rtol=1e-4
        
        # Verify preconditioner was initialized
        @test solver_prec.preconditioner_initialized[] == true
    end

    @testset "CG solvers with non-trivial diagonal preconditioner" begin
        nels = (8, 4, 4)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create a diagonal preconditioner with slightly varying diagonal values
        # This exercises the preconditioner update code path
        n_dofs = ndofs(problem.ch.dh)
        diag_vals = ones(Float64, n_dofs)
        # Add some variation while keeping it well-conditioned
        for i in 1:n_dofs
            diag_vals[i] = 1.0 + 0.1 * sin(i)
        end
        varying_prec = Preconditioners.DiagonalPreconditioner(diag_vals)

        # Test CGAssemblySolver
        solver_assembly = FEASolver(CGAssemblySolver, problem; abstol=1e-7, preconditioner=varying_prec)
        solver_assembly.vars .= 0.5
        solver_assembly()
        
        @test all(isfinite, solver_assembly.u)
        @test solver_assembly.preconditioner_initialized[] == true

        # Test CGMatrixFreeSolver
        # Need a fresh preconditioner for matrix-free solver
        varying_prec2 = Preconditioners.DiagonalPreconditioner(copy(diag_vals))
        solver_mf = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7, preconditioner=varying_prec2)
        solver_mf.vars .= 0.5
        solver_mf()
        
        @test all(isfinite, solver_mf.u)
        @test solver_mf.preconditioner_initialized[] == true
    end

    @testset "Preconditioner comparison with DirectSolver" begin
        nels = (6, 4, 4)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Direct solver (exact solution)
        solver_direct = FEASolver(DirectSolver, problem)
        solver_direct.vars .= 0.5
        solver_direct()

        # CG with preconditioner
        dummy_prec = Preconditioners.DiagonalPreconditioner(ones(Float64, ndofs(problem.ch.dh)))
        solver_cg = FEASolver(CGAssemblySolver, problem; abstol=1e-9, preconditioner=dummy_prec)
        solver_cg.vars .= 0.5
        solver_cg()

        # Results should be close to direct solver
        @test solver_cg.u ≈ solver_direct.u rtol=1e-6
    end
end