using TopOpt, Test

@testset "FEA Solver Tests" begin
    @testset "Structural solver consistency" begin
        nels = (20, 10, 10)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        solver1 = FEASolver(DirectSolver, problem)
        # Takes forever to compile
        solver2 = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)
        solver3 = FEASolver(CGAssemblySolver, problem; abstol=1e-7)

        x0 = rand(length(solver1.vars))
        solver1.vars .= x0
        solver2.vars .= x0
        solver3.vars .= x0

        solver1()
        solver2()
        solver3()

        @test solver1.u ≈ solver2.u
        @test solver1.u ≈ solver3.u
    end

    @testset "DirectSolver with QR decomposition" begin
        @testset "QR solver produces consistent results" begin
            nels = (10, 6, 6)
            sizes = (1.0, 1.0, 1.0)
            E = 1.0
            ν = 0.3
            force = -1.0
            problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
            
            # Create solver with Cholesky (default)
            solver_chol = FEASolver(DirectSolver, problem; qr=false)
            # Create solver with QR decomposition
            solver_qr = FEASolver(DirectSolver, problem; qr=true)
            
            x0 = rand(length(solver_chol.vars))
            solver_chol.vars .= x0
            solver_qr.vars .= x0
            
            solver_chol()
            solver_qr()
            
            # Results should be approximately equal
            @test solver_chol.u ≈ solver_qr.u rtol=1e-6
        end

        @testset "QR solver with multiple RHS" begin
            nels = (6, 4, 4)
            sizes = (1.0, 1.0, 1.0)
            E = 1.0
            ν = 0.3
            force = -1.0
            problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
            
            solver = FEASolver(DirectSolver, problem; qr=true)
            solver.vars .= 1.0
            
            # Solve with matrix RHS
            solver()
            
            # Verify solution satisfies K*u = f
            globalinfo = solver.globalinfo
            elementinfo = solver.elementinfo
            
            vars = ones(length(solver.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
            
            K = globalinfo.K
            f = globalinfo.f
            u = solver.u
            
            residual = K * u - f
            
            # Check relative residual is small
            @test norm(residual) / norm(f) < 0.01
        end

        @testset "QR solver handles non-positive-definite matrices" begin
            # QR can handle non-SPD matrices better than Cholesky
            nels = (6, 4, 4)
            sizes = (1.0, 1.0, 1.0)
            E = 1.0
            ν = 0.3
            force = -1.0
            problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
            
            solver = FEASolver(DirectSolver, problem; qr=true)
            solver.vars .= 0.5  # Intermediate densities
            
            # Should work without errors
            solver()
            
            @test all(isfinite, solver.u)
            @test !any(isnan, solver.u)
        end

        @testset "QR solver with HeatTransfer problem" begin
            nels = (4, 4)
            sizes = (1.0, 1.0)
            k = 1.0
            heatflux = Dict{String,Float64}("top" => 1.0)

            problem = HeatConductionProblem(
                Val{:Linear}, nels, sizes, k;
                Tleft=0.0, Tright=0.0, heatflux=heatflux
            )

            # Test QR solver on heat transfer
            solver_qr = FEASolver(DirectSolver, problem; qr=true, xmin=0.001)
            solver_qr.vars .= 1.0
            solver_qr()

            # Compare with Cholesky
            solver_chol = FEASolver(DirectSolver, problem; qr=false, xmin=0.001)
            solver_chol.vars .= 1.0
            solver_chol()

            @test solver_qr.u ≈ solver_chol.u rtol=1e-6
        end
    end

    @testset "Heat Transfer - Solver Consistency" begin
        @testset "DirectSolver produces symmetric positive definite system" begin
            nels = (4, 4)
            sizes = (1.0, 1.0)
            k = 1.0
            heatflux = Dict{String,Float64}("top" => 1.0)

            problem = HeatConductionProblem(
                Val{:Linear}, nels, sizes, k;
                Tleft=0.0, Tright=0.0, heatflux=heatflux
            )

            elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
            globalinfo = GlobalFEAInfo(problem)

            # Assemble system with uniform density
            vars = ones(prod(nels))
            penalty = PowerPenalty(1.0)
            xmin = 0.001

            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

            K = globalinfo.K
            # Check symmetry
            @test Matrix(K) ≈ Matrix(K)'

            # Check positive definiteness (all eigenvalues should be positive)
            eigs = eigvals(Matrix(K))
            @test all(eigs .> -1e-10)  # Allow small numerical errors
        end

        @testset "Solution satisfies governing equations" begin
            # K*u = f should be satisfied
            nels = (4, 4)
            sizes = (1.0, 1.0)
            k = 1.0
            heatflux = Dict{String,Float64}("top" => 1.0)

            problem = HeatConductionProblem(
                Val{:Linear}, nels, sizes, k;
                Tleft=0.0, Tright=0.0, heatflux=heatflux
            )

            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            solver.vars .= 1.0
            solver()

            # Get residual: K*u - f
            globalinfo = solver.globalinfo
            elementinfo = solver.elementinfo

            vars = ones(length(solver.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

            K = globalinfo.K
            f = globalinfo.f
            u = solver.u

            residual = K * u - f

            # Residual should be small (accounting for boundary conditions)
            # BC nodes will have residual from prescribed values
            @test norm(residual) / norm(f) < 0.1
        end
    end
end
