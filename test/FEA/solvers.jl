using TopOpt, Test, LinearAlgebra, Ferrite

@testset "FEA Solver Tests" begin
    @testset "Structural solver consistency" begin
        nels = (20, 10, 10)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create different solvers
        solver_direct = FEASolver(DirectSolver, problem)
        solver_assembly = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        solver_matrixfree = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)

        # Set variables to 0.5 (not 0 or 1)
        x0 = fill(0.5, length(solver_direct.vars))
        solver_direct.vars .= x0
        solver_assembly.vars .= x0
        solver_matrixfree.vars .= x0

        # Solve with each solver
        solver_direct()
        solver_assembly()
        solver_matrixfree()

        # Compare results - they should be very close for well-conditioned problems
        @test solver_assembly.u ≈ solver_direct.u rtol=1e-4
        @test solver_matrixfree.u ≈ solver_direct.u rtol=1e-4
    end

    @testset "CGAssemblySolver with safe=true" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        solver_safe = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        solver_normal = FEASolver(CGAssemblySolver, problem; abstol=1e-7)

        x0 = fill(0.5, length(solver_safe.vars))
        solver_safe.vars .= x0
        solver_normal.vars .= x0

        # Solve with safe=true and safe=false
        solver_safe(false, Val{true})
        solver_normal(false, Val{false})

        @test all(isfinite, solver_safe.u)
        @test all(isfinite, solver_normal.u)
        @test solver_safe.u ≈ solver_normal.u rtol=1e-4
    end

    @testset "DirectSolver with QR decomposition" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test multiple RHS with DirectSolver
        num_rhs = 3
        solver = FEASolver(DirectSolver, problem)
        solver.vars .= 1.0

        ndofs = length(solver.u)
        rhs_matrix = rand(ndofs, num_rhs)
        lhs_matrix = similar(rhs_matrix)

        # Solve with QR
        solver(; solver=Val{:QR}, rhs=rhs_matrix, lhs=lhs_matrix)

        # Verify solutions
        for j in 1:num_rhs
            @test all(isfinite, lhs_matrix[:, j])
        end
    end

    @testset "Show methods for GenericFEASolver" begin
        nels = (8, 4, 4)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test that show methods don't error
        solver_direct = FEASolver(DirectSolver, problem)
        @test_nowarn show(devnull, MIME("text/plain"), solver_direct)

        solver_assembly = FEASolver(CGAssemblySolver, problem)
        @test_nowarn show(devnull, MIME("text/plain"), solver_assembly)

        solver_matrixfree = FEASolver(CGMatrixFreeSolver, problem)
        @test_nowarn show(devnull, MIME("text/plain"), solver_matrixfree)

        # Test default show (compact)
        @test_nowarn show(devnull, solver_direct)
        @test_nowarn show(devnull, solver_assembly)
        @test_nowarn show(devnull, solver_matrixfree)
    end

    @testset "GenericFEASolver with matrix RHS and assemble_f=true" begin
        nels = (10, 6, 6)
        sizes = (1.0, 1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = -1.0
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        num_rhs = 3

        @testset "DirectSolver with matrix RHS" begin
            solver = FEASolver(DirectSolver, problem)
            solver.vars .= 1.0

            ndofs = length(solver.u)
            rhs_matrix = rand(ndofs, num_rhs)
            lhs_matrix = similar(rhs_matrix)

            # Assemble K and get a reference solution for comparison
            globalinfo = solver.globalinfo
            elementinfo = solver.elementinfo
            vars = ones(length(solver.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
            K = globalinfo.K

            # Get reference solutions by solving each column with boundary conditions applied
            lhs_reference = similar(rhs_matrix, Float64)
            for j in 1:num_rhs
                rhs_j = copy(rhs_matrix[:, j])
                Ferrite.apply_zero!(rhs_j, problem.ch)
                lhs_reference[:, j] = K \ rhs_j
            end

            # Now call the solver with matrix RHS and assemble_f=true
            # The assembled f will be used for the first solve, then RHS columns for subsequent solves
            solver(; assemble_f=true, rhs=rhs_matrix, lhs=lhs_matrix)

            # Verify solutions match reference
            for j in 1:num_rhs
                @test lhs_matrix[:, j] ≈ lhs_reference[:, j] rtol=1e-4
            end
        end

        @testset "CGAssemblySolver with matrix RHS" begin
            solver = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
            solver.vars .= 1.0

            ndofs = length(solver.u)
            rhs_matrix = rand(ndofs, num_rhs)
            lhs_matrix = similar(rhs_matrix)

            # Assemble K
            globalinfo = solver.globalinfo
            elementinfo = solver.elementinfo
            vars = ones(length(solver.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
            K = globalinfo.K

            # Get reference solutions
            lhs_reference = similar(rhs_matrix, Float64)
            for j in 1:num_rhs
                rhs_j = copy(rhs_matrix[:, j])
                Ferrite.apply_zero!(rhs_j, problem.ch)
                lhs_reference[:, j] = K \ rhs_j
            end

            solver(; assemble_f=true, rhs=rhs_matrix, lhs=lhs_matrix)

            for j in 1:num_rhs
                @test lhs_matrix[:, j] ≈ lhs_reference[:, j] rtol=1e-4
            end
        end

        @testset "CGMatrixFreeSolver with matrix RHS" begin
            solver = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)
            solver.vars .= 1.0

            ndofs = length(solver.u)
            rhs_matrix = rand(ndofs, num_rhs)
            lhs_matrix = similar(rhs_matrix)

            # Get reference solutions using DirectSolver
            solver_ref = FEASolver(DirectSolver, problem)
            solver_ref.vars .= 1.0
            globalinfo = solver_ref.globalinfo
            elementinfo = solver_ref.elementinfo
            vars = ones(length(solver_ref.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
            K = globalinfo.K

            lhs_reference = similar(rhs_matrix, Float64)
            for j in 1:num_rhs
                rhs_j = copy(rhs_matrix[:, j])
                Ferrite.apply_zero!(rhs_j, problem.ch)
                lhs_reference[:, j] = K \ rhs_j
            end

            solver(; assemble_f=true, rhs=rhs_matrix, lhs=lhs_matrix)

            for j in 1:num_rhs
                @test lhs_matrix[:, j] ≈ lhs_reference[:, j] rtol=1e-3
            end
        end

        @testset "HeatTransfer with matrix RHS" begin
            nels = (4, 4)
            sizes = (1.0, 1.0)
            k = 1.0
            heatflux = Dict{String,Float64}("top" => 1.0)
            problem = HeatConductionProblem(
                Val{:Linear}, nels, sizes, k;
                Tleft=0.0, Tright=0.0, heatflux=heatflux
            )

            solver = FEASolver(DirectSolver, problem)
            solver.vars .= 1.0

            ndofs = length(solver.u)
            rhs_matrix = rand(ndofs, num_rhs)
            lhs_matrix = similar(rhs_matrix)

            # Get reference solutions
            globalinfo = solver.globalinfo
            elementinfo = solver.elementinfo
            vars = ones(length(solver.vars))
            penalty = PowerPenalty(1.0)
            xmin = 0.001
            TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
            K = globalinfo.K

            lhs_reference = similar(rhs_matrix, Float64)
            for j in 1:num_rhs
                rhs_j = copy(rhs_matrix[:, j])
                Ferrite.apply_zero!(rhs_j, problem.ch)
                lhs_reference[:, j] = K \ rhs_j
            end

            solver(; assemble_f=true, rhs=rhs_matrix, lhs=lhs_matrix)

            for j in 1:num_rhs
                @test lhs_matrix[:, j] ≈ lhs_reference[:, j] rtol=1e-4
            end
        end

        @testset "Matrix RHS consistency across solvers" begin
            ndofs = let
                s = FEASolver(DirectSolver, problem)
                length(s.u)
            end
            rhs_matrix = rand(ndofs, num_rhs)

            solvers = [
                ("DirectSolver", FEASolver(DirectSolver, problem)),
                ("CGAssemblySolver", FEASolver(CGAssemblySolver, problem; abstol=1e-7)),
                ("CGMatrixFreeSolver", FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)),
            ]

            results = Dict()
            for (name, solver) in solvers
                solver.vars .= 1.0
                lhs = similar(rhs_matrix)
                solver(; assemble_f=true, rhs=rhs_matrix, lhs=lhs)
                results[name] = lhs
            end

            @test results["CGAssemblySolver"] ≈ results["DirectSolver"] rtol=1e-4
            @test results["CGMatrixFreeSolver"] ≈ results["DirectSolver"] rtol=1e-3
        end
    end

    @testset "Heat Transfer - Solver Consistency" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        # Create different solvers for heat transfer
        solver_direct = FEASolver(DirectSolver, problem)
        solver_assembly = FEASolver(CGAssemblySolver, problem; abstol=1e-7)
        solver_matrixfree = FEASolver(CGMatrixFreeSolver, problem; abstol=1e-7)

        # Set variables
        x0 = fill(0.5, length(solver_direct.vars))
        solver_direct.vars .= x0
        solver_assembly.vars .= x0
        solver_matrixfree.vars .= x0

        # Solve
        solver_direct()
        solver_assembly()
        solver_matrixfree()

        # Compare results
        @test solver_assembly.u ≈ solver_direct.u rtol=1e-4
        @test solver_matrixfree.u ≈ solver_direct.u rtol=1e-4
    end
end
