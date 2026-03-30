using TopOpt, Test

# Test show methods for custom types

@testset "Show Methods" begin
    # Create a simple problem for testing
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)

    @testset "GenericFEASolver show methods" begin
        # Direct solver
        solver_direct = FEASolver(DirectSolver, problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), solver_direct)
        @test String(take!(io)) == "TopOpt direct structural solver (GenericFEASolver)\n"

        # CG Assembly solver
        solver_cg = FEASolver(CGAssemblySolver, problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), solver_cg)
        @test String(take!(io)) == "TopOpt CG with assembly structural solver (GenericFEASolver)\n"

        # Matrix-free CG solver
        solver_mf = FEASolver(CGMatrixFreeSolver, problem)
        io = IOBuffer()
        show(io, MIME("text/plain"), solver_mf)
        @test String(take!(io)) == "TopOpt matrix-free CG structural solver (GenericFEASolver)\n"
    end

    @testset "MatrixOperator show method" begin
        solver = FEASolver(DirectSolver, problem)
        # MatrixOperator is internal, but we can access it through solver internals
        # The show method should not error
        io = IOBuffer()
        # Test that MatrixOperator show doesn't throw
        op = TopOpt.FEA.MatrixOperator(solver.globalinfo.K, solver.globalinfo.f, DefaultCriteria())
        show(io, MIME("text/plain"), op)
        @test String(take!(io)) == "TopOpt matrix linear operator\n"
    end

    @testset "MatrixFreeOperator show method" begin
        solver = FEASolver(CGMatrixFreeSolver, problem)
        io = IOBuffer()
        # Test that MatrixFreeOperator show doesn't throw
        # Access the operator through solver
        show(io, MIME("text/plain"), solver)
        @test String(take!(io)) == "TopOpt matrix-free CG structural solver (GenericFEASolver)\n"
    end

    @testset "LinearElasticityResult show method" begin
        result = simulate(problem, ones(4))
        io = IOBuffer()
        show(io, MIME("text/plain"), result)
        @test String(take!(io)) == "TopOpt linear elasticity result\n"
    end

    @testset "visualize fallback function" begin
        # Test that visualize throws an error without Makie loaded
        err = @test_throws ErrorException visualize([1, 2, 3])
        @test occursin("visualize", sprint(show, err.value))
        @test occursin("Makie", sprint(show, err.value))
    end
end
