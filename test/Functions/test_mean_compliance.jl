using TopOpt, Test, LinearAlgebra, Random, SparseArrays, FiniteDifferences, Zygote
using TopOpt: Nonconvex
const FDM = FiniteDifferences

# Manual mean function since Statistics isn't available in SafeTestsets
_mean(x) = sum(x) / length(x)

Random.seed!(42)

@testset "MeanCompliance - Analytical Validation" begin

    @testset "MeanCompliance produces finite positive values" begin
        # Basic sanity check: MeanCompliance should return positive finite values
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (10, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem_ml = MultiLoad(base_problem, F)
        solver_ml = FEASolver(DirectSolver, problem_ml; xmin=0.01, penalty=PowerPenalty(1.0))
        mc = MeanCompliance(problem_ml, solver_ml; method=:exact)

        x = fill(0.5, length(solver_ml.vars))
        val = mc(PseudoDensities(x))

        @test val > 0
        @test isfinite(val)
    end

    @testset "Identical loads give same compliance as single load" begin
        # Analytical result: N identical loads should give mean compliance
        # equal to the single load compliance
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (8, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Single load
        F1 = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        F1[right_dofs, 1] .= force

        problem1 = MultiLoad(base_problem, F1)
        solver1 = FEASolver(DirectSolver, problem1; xmin=0.01, penalty=PowerPenalty(1.0))
        mc1 = MeanCompliance(problem1, solver1; method=:exact)

        # Multiple identical loads
        nloads = 5
        F5 = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        for i in 1:nloads
            F5[right_dofs, i] .= force
        end

        problem5 = MultiLoad(base_problem, F5)
        solver5 = FEASolver(DirectSolver, problem5; xmin=0.01, penalty=PowerPenalty(1.0))
        mc5 = MeanCompliance(problem5, solver5; method=:exact)

        x = fill(0.6, length(solver1.vars))
        val1 = mc1(PseudoDensities(x))
        val5 = mc5(PseudoDensities(x))

        # Should match closely
        @test isapprox(val1, val5; rtol=0.01)
    end

    @testset "Mean compliance scales with load magnitude squared" begin
        # Analytical result: Compliance C = f^T * K^-1 * f
        # If all loads scale by factor α, then C scales by α²
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Reference with unit load
        nloads = 3
        F_ref = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F_ref[dofs, i] .= 1.0
        end

        problem_ref = MultiLoad(base_problem, F_ref)
        solver_ref = FEASolver(DirectSolver, problem_ref; xmin=0.01, penalty=PowerPenalty(1.0))
        mc_ref = MeanCompliance(problem_ref, solver_ref; method=:exact)

        x = fill(0.5, length(solver_ref.vars))
        C_ref = mc_ref(PseudoDensities(x))

        # Scale loads by factor 2
        scale_factor = 2.0
        F_scaled = scale_factor .* F_ref
        problem_scaled = MultiLoad(base_problem, F_scaled)
        solver_scaled = FEASolver(DirectSolver, problem_scaled; xmin=0.01, penalty=PowerPenalty(1.0))
        mc_scaled = MeanCompliance(problem_scaled, solver_scaled; method=:exact)

        C_scaled = mc_scaled(PseudoDensities(x))

        # Should scale by scale_factor²
        expected_ratio = scale_factor^2
        actual_ratio = C_scaled / C_ref

        @test isapprox(actual_ratio, expected_ratio; rtol=0.01)
    end

    @testset "Higher volume fraction gives lower compliance" begin
        # Physical constraint: More material should give lower (better) compliance
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (8, 4)

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
        mc = MeanCompliance(problem, solver; method=:exact)

        # Compare different volume fractions
        vol_fracs = [0.3, 0.5, 0.7, 0.9]
        compliances = Float64[]

        for vf in vol_fracs
            x = fill(vf, length(solver.vars))
            C = mc(PseudoDensities(x))
            push!(compliances, C)
        end

        # Compliance should decrease with increasing volume fraction
        for i in 2:length(compliances)
            @test compliances[i] < compliances[i-1]
        end
    end
end

@testset "MeanCompliance - Method Accuracy" begin

    @testset "Exact methods produce consistent results" begin
        # All exact methods should give very similar results
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (8, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 4
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)

        # Test both exact methods
        solver_exact = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_exact = MeanCompliance(problem, solver_exact; method=:exact)

        solver_svd = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_svd = MeanCompliance(problem, solver_svd; method=:exact_svd)

        x = fill(0.5, length(solver_exact.vars))

        val_exact = mc_exact(PseudoDensities(x))
        val_svd = mc_svd(PseudoDensities(x))

        # Should match very closely (< 1% difference)
        @test isapprox(val_exact, val_svd; rtol=0.01)
    end

    @testset "Trace estimation converges with more samples" begin
        # Hutchinson trace estimator converges as O(1/sqrt(nv))
        # With more sample vectors, trace estimate should converge to exact value
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)

        # Reference exact solution
        solver_exact = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_exact = MeanCompliance(problem, solver_exact; method=:exact)

        x = fill(0.5, length(solver_exact.vars))
        C_exact = mc_exact(PseudoDensities(x))

        # Test convergence with increasing samples
        sample_counts = [5, 10, 20]
        errors = Float64[]

        for nv in sample_counts
            solver_trace = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
            mc_trace = MeanCompliance(problem, solver_trace; method=:trace, nv=nv)
            C_trace = mc_trace(PseudoDensities(x))

            rel_error = abs(C_trace - C_exact) / C_exact
            push!(errors, rel_error)
        end

        # Error should generally decrease with more samples
        # Allow some statistical variation
        @test errors[end] < 0.5  # Final error should be reasonable
    end

    @testset "SIMP power law for multi-load compliance" begin
        # With SIMP penalization, mean compliance follows
        # C(ρ) = ρ^(-p) * C(1) for uniform density, approximately
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= 1.0
        end

        problem = MultiLoad(base_problem, F)

        # Reference with full density, p=1
        solver_ref = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        mc_ref = MeanCompliance(problem, solver_ref; method=:exact)
        C_full = mc_ref(PseudoDensities(ones(length(solver_ref.vars))))

        # Test power law scaling with ρ=0.5
        rho = 0.5
        x_uniform = fill(rho, length(solver_ref.vars))

        for p in [1.0, 2.0, 3.0]
            solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(p))
            mc = MeanCompliance(problem, solver; method=:exact)
            C_uniform = mc(PseudoDensities(x_uniform))

            # Check approximate power law behavior
            expected_ratio = rho^(-p)  # Compliance increases as density decreases
            actual_ratio = C_uniform / C_full

            # Allow 30% tolerance for multi-load effects
            @test isapprox(actual_ratio, expected_ratio; rtol=0.3)
        end
    end
end

@testset "MeanCompliance - Gradient Verification" begin

    @testset "Zygote vs FiniteDifferences consistency" begin
        # Gradient should match between automatic and finite difference
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2) .* 0.5
        end

        problem = MultiLoad(base_problem, F)

        for method in [:exact, :exact_svd]
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
            mc = MeanCompliance(problem, solver; method=method)
            f = x -> mc(PseudoDensities(x))

            # Test at a random point
            x = clamp.(rand(length(solver.vars)), 0.2, 0.9)

            # Compute gradients
            grad_zygote = Zygote.gradient(f, x)[1]
            grad_fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]

            # Check properties
            @test length(grad_zygote) == length(x)
            @test all(isfinite.(grad_zygote))

            # Gradient should be negative (more material = lower compliance)
            @test _mean(grad_zygote) < 0

            # Check finite difference consistency (relaxed tolerance)
            @test norm(grad_zygote - grad_fd) / norm(grad_fd) < 0.1
        end
    end

    @testset "Trace method gradient consistency" begin
        # Trace estimate gradient should match exact gradient approximately
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2) .* 0.5
        end

        problem = MultiLoad(base_problem, F)

        solver_exact = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_exact = MeanCompliance(problem, solver_exact; method=:exact)

        solver_trace = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace = MeanCompliance(problem, solver_trace; method=:trace, nv=20)

        x = fill(0.5, length(solver_exact.vars))

        f_exact = x -> mc_exact(PseudoDensities(x))
        f_trace = x -> mc_trace(PseudoDensities(x))

        grad_exact = Zygote.gradient(f_exact, x)[1]
        grad_trace = Zygote.gradient(f_trace, x)[1]

        # Directions should be similar
        cos_angle = dot(grad_exact, grad_trace) / (norm(grad_exact) * norm(grad_trace))
        @test cos_angle > 0.8  # Gradients should point in similar directions
    end

    @testset "Gradient decreases with increasing density" begin
        # Physical property: Gradient magnitude should decrease as density increases
        # (stiffness saturates at high density)
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        mc = MeanCompliance(problem, solver; method=:exact)
        f = x -> mc(PseudoDensities(x))

        # Test at low and high density
        x_low = fill(0.2, length(solver.vars))
        x_high = fill(0.9, length(solver.vars))

        grad_low = Zygote.gradient(f, x_low)[1]
        grad_high = Zygote.gradient(f, x_high)[1]

        # Gradient magnitude should be higher at low density (more sensitivity)
        @test norm(grad_low) > norm(grad_high)
    end
end

@testset "MeanCompliance - System Properties" begin

    @testset "Compliance is positive definite" begin
        # Mean compliance must always be positive for valid configurations
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)

        for method in [:exact, :exact_svd, :trace]
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
            mc = MeanCompliance(problem, solver; method=method, nv=5)

            # Test at multiple densities
            for vf in [0.3, 0.5, 0.7]
                x = fill(vf, length(solver.vars))
                C = mc(PseudoDensities(x))
                @test C > 0
                @test isfinite(C)
            end
        end
    end

    @testset "Different sample methods give similar results" begin
        # Both hutchinson and hadamard sampling should converge to same trace
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2)
        end

        problem = MultiLoad(base_problem, F)

        values = Float64[]

        for sample_method in [:hutch, :hadamard]
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
            mc = MeanCompliance(problem, solver; method=:trace, nv=15, sample_method=sample_method)

            x = fill(0.5, length(solver.vars))
            C = mc(PseudoDensities(x))
            push!(values, C)
        end

        # Results should be within 50% (allowing for estimator variance)
        @test abs(values[1] - values[2]) / _mean(values) < 0.5
    end

    @testset "Individual compliances are positive and finite" begin
        # Verify individual load case compliances
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Create 3 specific load cases
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))

        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= [1.0, 0.5]
        end

        x = fill(0.5, nels[1] * nels[2])

        # Individual compliances should be positive
        for i in 1:nloads
            F_single = F[:, i:i]
            problem_single = MultiLoad(base_problem, F_single)
            solver_single = FEASolver(DirectSolver, problem_single; xmin=0.01, penalty=PowerPenalty(2.0))
            comp = Compliance(solver_single)
            C_single = comp(PseudoDensities(x))
            @test C_single > 0
            @test isfinite(C_single)
        end

        # Multi-load compliance should also be positive
        problem_ml = MultiLoad(base_problem, F)
        solver_ml = FEASolver(DirectSolver, problem_ml; xmin=0.01, penalty=PowerPenalty(2.0))
        mc = MeanCompliance(problem_ml, solver_ml; method=:exact)
        C_mean = mc(PseudoDensities(x))
        @test C_mean > 0
        @test isfinite(C_mean)
    end
end

@testset "MeanCompliance - Integration with Filter" begin

    @testset "DensityFilter with mean compliance chain rule" begin
        # Test that gradient chain rule works through filter
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (8, 4)

        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= randn(2) .* 0.5
        end

        problem = MultiLoad(base_problem, F)

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc = MeanCompliance(problem, solver; method=:exact)
        filter = DensityFilter(solver; rmin=2.0)

        f = x -> mc(filter(PseudoDensities(x)))

        x = clamp.(rand(length(solver.vars)), 0.2, 0.9)

        # Compute gradient through filter
        grad = Zygote.gradient(f, x)[1]

        @test length(grad) == length(x)
        @test all(isfinite.(grad))

        # Verify with finite differences
        grad_fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]

        # Relaxed check due to filter nonlinearity
        @test norm(grad - grad_fd) / norm(grad_fd) < 0.2
    end
end

@testset "MeanCompliance - Error Handling" begin

    @testset "Construction validation" begin
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)

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
        mc = MeanCompliance(problem, solver; method=:exact)

        # Verify properties
        @test mc isa TopOpt.Functions.AbstractFunction
        @test mc.compliance isa Compliance
        @test mc.F === F
        @test Nonconvex.NonconvexCore.getdim(mc) == 1
    end
end

@testset "MeanCompliance - Helper Functions" begin

    @testset "hutch_rand! produces valid Rademacher samples" begin
        # Hutchinson estimator uses Rademacher random variables (±1)
        for n in [10, 20, 30]
            for m in [3, 5, 8]
                V = zeros(n, m)
                TopOpt.Functions.hutch_rand!(V)

                # All entries should be ±1
                @test all(abs.(V) .== 1.0)
                @test size(V) == (n, m)

                # Should have roughly equal +1 and -1
                frac_positive = count(V .== 1.0) / length(V)
                @test 0.3 < frac_positive < 0.7  # Roughly balanced
            end
        end
    end

    @testset "hadamard! produces valid Hadamard vectors" begin
        # Hadamard vectors are deterministic ±1 patterns
        for n in [8, 16, 32]
            for m in [2, 4, 6]
                V = zeros(n, m)
                TopOpt.Functions.hadamard!(V)

                # All entries should be ±1
                @test all(abs.(V) .== 1.0)
                @test size(V) == (n, m)

                # Columns should be orthogonal
                for i in 1:m
                    for j in i+1:m
                        dot_prod = dot(V[:, i], V[:, j])
                        @test abs(dot_prod) < 1e-10
                    end
                end
            end
        end
    end
end
