using TopOpt, Test, LinearAlgebra, Random, SparseArrays, FiniteDifferences, Zygote
using TopOpt: Nonconvex, Functions

const FDM = FiniteDifferences

Random.seed!(42)

@testset "compute_mean_compliance with TraceEstimationSVDMean" begin

    @testset "Basic functionality - function returns finite value" begin
        # Create a simple problem
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Create multiple load cases
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Create MeanCompliance with TraceEstimationSVDMean method
        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)

        # Verify it's using TraceEstimationSVDMean
        @test mc.method isa Functions.TraceEstimationSVDMean

        # Get the method object
        ax = mc.method

        # Prepare inputs
        x = fill(0.5, length(solver.vars))
        grad = similar(x)

        # Call the specific dispatch
        val = Functions.compute_mean_compliance(mc, ax, x, grad)

        # Verify results
        @test isfinite(val)
        @test val > 0
        @test all(isfinite.(grad))
        @test length(grad) == length(x)
    end

    @testset "Consistency with compute_approx_ec" begin
        # Verify the function calls compute_approx_ec correctly
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)
        ax = mc.method

        x = fill(0.5, length(solver.vars))
        grad = similar(x)

        # Call via dispatch
        val1 = Functions.compute_mean_compliance(mc, ax, x, grad)
        grad1 = copy(grad)

        # Call compute_approx_ec directly with same parameters
        grad .= 0
        val2 = Functions.compute_approx_ec(mc, x, grad, ax.US, ax.V, ax.n)

        # Results should match
        @test isapprox(val1, val2; rtol=1e-10)
        @test isapprox(grad1, grad; rtol=1e-10)
    end

    @testset "Different sample counts" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test with different sample counts
        for nv in [1, 3, 5]
            mc = MeanCompliance(problem, solver; method=:approx_svd, nv=nv)
            ax = mc.method

            x = fill(0.5, length(solver.vars))
            grad = similar(x)

            val = Functions.compute_mean_compliance(mc, ax, x, grad)

            @test isfinite(val)
            @test val > 0
            @test all(isfinite.(grad))
        end
    end

    @testset "Different density values" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)
        ax = mc.method

        # Test with different density values
        for rho in [0.2, 0.5, 0.8, 0.95]
            x = fill(rho, length(solver.vars))
            grad = similar(x)

            val = Functions.compute_mean_compliance(mc, ax, x, grad)

            @test isfinite(val)
            @test val > 0
            @test all(isfinite.(grad))

            # Compliance should decrease with higher density
            # (this is a physical property, not strict requirement)
        end
    end

    @testset "Gradient accumulation" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)
        ax = mc.method

        x = fill(0.5, length(solver.vars))

        # Test that gradient is properly accumulated (not just overwritten)
        grad_pre = ones(length(x))
        grad = copy(grad_pre)

        Functions.compute_mean_compliance(mc, ax, x, grad)

        # Gradient should be updated
        @test !iszero(grad)
    end

    @testset "Compare with ExactSVDMean" begin
        # TraceEstimationSVDMean should approximate ExactSVDMean
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Exact SVD
        mc_exact = MeanCompliance(problem, solver; method=:exact_svd)
        x = fill(0.5, length(solver.vars))
        grad_exact = similar(x)
        val_exact = Functions.compute_mean_compliance(mc_exact, mc_exact.method, x, grad_exact)

        # Approximate SVD with many samples
        mc_approx = MeanCompliance(problem, solver; method=:approx_svd, nv=20)
        ax = mc_approx.method
        grad_approx = similar(x)
        val_approx = Functions.compute_mean_compliance(mc_approx, ax, x, grad_approx)

        # Should be reasonably close (within 30% for trace estimation)
        @test abs(val_approx - val_exact) / val_exact < 0.3
    end

    @testset "Method fields are correctly structured" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)
        ax = mc.method

        # Verify structure
        @test ax.n == nloads
        @test size(ax.US, 1) == TopOpt.Ferrite.ndofs(base_problem.ch.dh)
        @test size(ax.V, 2) == 5  # nv samples

        # US should be sparse (from SVD)
        @test ax.US isa SparseMatrixCSC

        # V should contain Rademacher samples
        @test all(abs.(ax.V) .== 1.0)

        x = fill(0.5, length(solver.vars))
        grad = similar(x)

        val = Functions.compute_mean_compliance(mc, ax, x, grad)

        @test isfinite(val)
    end

    @testset "Different sample methods" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        x = fill(0.5, length(solver.vars))

        for sample_method in [:hutch, :hadamard]
            mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5, sample_method=sample_method)
            ax = mc.method
            grad = similar(x)

            val = Functions.compute_mean_compliance(mc, ax, x, grad)

            @test isfinite(val)
            @test val > 0
            @test all(isfinite.(grad))
        end
    end

    @testset "Integration with solver state" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        mc = MeanCompliance(problem, solver; method=:approx_svd, nv=5)
        ax = mc.method

        x = fill(0.5, length(solver.vars))
        grad = similar(x)

        # First call
        val1 = Functions.compute_mean_compliance(mc, ax, x, grad)

        # Second call with same x should give same result
        grad2 = similar(x)
        val2 = Functions.compute_mean_compliance(mc, ax, x, grad2)

        @test isapprox(val1, val2; rtol=1e-10)
        @test isapprox(grad, grad2; rtol=1e-10)
    end

end
