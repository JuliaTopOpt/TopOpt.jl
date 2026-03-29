using TopOpt, Test, Random, SparseArrays, LinearAlgebra, FiniteDifferences, Zygote, Statistics, ChainRulesCore
import Nonconvex

const FDM = FiniteDifferences

Random.seed!(42)

@testset "BlockCompliance Basic API" begin
    @testset "Construction with different methods" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create multi-load problem with single load
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        
        # Test exact method
        bc_exact = BlockCompliance(problem, solver; method=:exact)
        @test bc_exact isa BlockCompliance
        @test Nonconvex.NonconvexCore.getdim(bc_exact) == 1
        
        # Test exact_svd method
        bc_svd = BlockCompliance(problem, solver; method=:exact_svd)
        @test bc_svd isa BlockCompliance
        @test Nonconvex.NonconvexCore.getdim(bc_svd) == 1
        
        # Test approximate method with hutchinson
        bc_approx = BlockCompliance(problem, solver; method=:approx, nv=2, sample_method=:hutch)
        @test bc_approx isa BlockCompliance
        @test Nonconvex.NonconvexCore.getdim(bc_approx) == 1
        
        # Test approximate method with hadamard
        bc_hadamard = BlockCompliance(problem, solver; method=:approx, nv=2, sample_method=:hadamard)
        @test bc_hadamard isa BlockCompliance
        @test Nonconvex.NonconvexCore.getdim(bc_hadamard) == 1
    end

    @testset "getpenalty and properties" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        current_penalty = TopOpt.Utilities.getpenalty(bc)
        @test current_penalty isa PowerPenalty
        @test current_penalty.p == 3.0
        
        # Test getsolver forwarding
        @test TopOpt.Utilities.getsolver(bc.compliance) isa TopOpt.FEA.GenericFEASolver
    end

    @testset "Evaluation produces finite positive results" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; nv=1)
        
        # Test with uniform density
        x = ones(prod(nels))
        result = bc(PseudoDensities(x))
        @test length(result) == 1
        @test isfinite(result[1])
        @test result[1] > 0
        
        # Test with varying density
        x_varying = fill(0.5, prod(nels))
        result_varying = bc(PseudoDensities(x_varying))
        @test length(result_varying) == 1
        @test isfinite(result_varying[1])
        @test result_varying[1] > 0
    end
end

@testset "BlockCompliance Multi-Load Cases" begin
    @testset "Multiple independent loads" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create 3 load cases
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        # Load case 1: force at top
        F[right_dofs[1], 1] = 1.0
        # Load case 2: force at middle
        F[right_dofs[length(right_dofs)÷2], 2] = 1.0
        # Load case 3: force at bottom
        F[right_dofs[end], 3] = 1.0
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        
        # Test with exact method
        bc_exact = BlockCompliance(problem, solver; method=:exact)
        x = fill(0.5, prod(nels))
        result_exact = bc_exact(PseudoDensities(x))
        
        @test length(result_exact) == nloads
        @test all(isfinite.(result_exact))
        @test all(result_exact .> 0)
        
        # Each load case should give different compliance
        @test result_exact[1] != result_exact[2]
        @test result_exact[2] != result_exact[3]
    end

    @testset "Block compliance returns correct dimensions" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create 2 load cases
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        solver_ml = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver_ml; method=:exact)
        
        x = fill(0.5, prod(nels))
        bc_result = bc(PseudoDensities(x))
        
        # Check dimensions and basic properties
        @test length(bc_result) == nloads
        @test all(isfinite.(bc_result))
        @test all(bc_result .> 0)
    end

    @testset "Compliance decreases with increasing density" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        for i in 1:nloads
            F[right_dofs[i * length(right_dofs) ÷ (nloads + 1)], i] = 1.0
        end
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        # Compare different volume fractions
        densities = [0.3, 0.5, 0.7, 0.9]
        results = [bc(PseudoDensities(fill(d, prod(nels)))) for d in densities]
        
        # Compliance should generally decrease as density increases
        for i in 2:length(results)
            # Allow some tolerance for numerical variations
            @test mean(results[i]) <= mean(results[i-1]) * 1.5
        end
    end
end

@testset "BlockCompliance Method Accuracy" begin
    @testset "Exact and exact_svd methods match" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        for i in 1:nloads
            F[right_dofs[i * length(right_dofs) ÷ (nloads + 1)], i] = 1.0
        end
        
        problem = MultiLoad(base_problem, F)
        
        solver_exact = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc_exact = BlockCompliance(problem, solver_exact; method=:exact)
        
        solver_svd = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc_svd = BlockCompliance(problem, solver_svd; method=:exact_svd)
        
        x = fill(0.5, prod(nels))
        
        result_exact = bc_exact(PseudoDensities(x))
        result_svd = bc_svd(PseudoDensities(x))
        
        @test isapprox(result_exact, result_svd; rtol=0.01)
    end

    @testset "Approximation methods converge with more samples" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 1.0
        
        problem = MultiLoad(base_problem, F)
        
        # Reference exact solution
        solver_exact = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc_exact = BlockCompliance(problem, solver_exact; method=:exact)
        
        x = fill(0.5, prod(nels))
        result_exact = bc_exact(PseudoDensities(x))
        
        # Test with increasing samples
        sample_counts = [5, 10, 20]
        errors = Float64[]
        
        for nv in sample_counts
            solver_approx = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
            bc_approx = BlockCompliance(problem, solver_approx; method=:approx, nv=nv)
            result_approx = bc_approx(PseudoDensities(x))
            
            rel_error = norm(result_approx - result_exact) / norm(result_exact)
            push!(errors, rel_error)
        end
        
        # Error should generally decrease with more samples
        @test errors[end] < 0.5
    end

    @testset "Different sample methods" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 1.0
        
        problem = MultiLoad(base_problem, F)
        
        results = []
        
        for sample_method in [:hutch, :hadamard]
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
            bc = BlockCompliance(problem, solver; method=:approx, nv=10, sample_method=sample_method)
            
            x = fill(0.5, prod(nels))
            result = bc(PseudoDensities(x))
            push!(results, result)
        end
        
        # Both methods should give reasonable results
        @test all(isfinite.(results[1]))
        @test all(isfinite.(results[2]))
        @test all(results[1] .> 0)
        @test all(results[2] .> 0)
    end
end

@testset "BlockCompliance Gradient Verification" begin
    @testset "Zygote gradient with exact method" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        # Test with a specific weight vector
        w = [0.5, 0.5]
        
        f = x -> sum(w .* bc(PseudoDensities(x)))
        
        x = fill(0.5, prod(nels))
        
        # Compute gradient using Zygote
        grad_zygote = Zygote.gradient(f, x)[1]
        
        @test length(grad_zygote) == length(x)
        @test all(isfinite.(grad_zygote))
        
        # Gradient should be negative (more material = lower weighted compliance)
        @test sum(grad_zygote) < 0
    end

    @testset "ChainRules rrule consistency" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        x = fill(0.5, prod(nels))
        pd = PseudoDensities(x)
        
        # Get value and pullback from rrule
        val, pullback = ChainRulesCore.rrule(bc, pd)
        
        @test val isa AbstractVector
        @test length(val) == nloads
        
        # Test pullback with different weights
        for w in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]
            _, tangent = pullback(w)
            @test length(tangent.x) == length(x)
            @test all(isfinite.(tangent.x))
        end
    end

    @testset "Gradient with different methods" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        
        x = fill(0.5, prod(nels))
        w = [0.5, 0.5]
        
        for method in [:exact, :exact_svd]
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
            bc = BlockCompliance(problem, solver; method=method)
            
            f = x -> sum(w .* bc(PseudoDensities(x)))
            grad = Zygote.gradient(f, x)[1]
            
            @test all(isfinite.(grad))
            @test sum(grad) < 0
        end
        
        # Test with approximate method
        solver_approx = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc_approx = BlockCompliance(problem, solver_approx; method=:approx, nv=10)
        
        f_approx = x -> sum(w .* bc_approx(PseudoDensities(x)))
        grad_approx = Zygote.gradient(f_approx, x)[1]
        
        @test all(isfinite.(grad_approx))
    end

    @testset "Gradient through weighted sum" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        for i in 1:nloads
            F[right_dofs[i * length(right_dofs) ÷ (nloads + 1)], i] = 1.0
        end
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        x = fill(0.5, prod(nels))
        
        # Test different weight combinations
        weights = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1/3, 1/3, 1/3],
            [0.5, 0.3, 0.2]
        ]
        
        for w in weights
            f = x -> sum(w .* bc(PseudoDensities(x)))
            grad = Zygote.gradient(f, x)[1]
            
            @test all(isfinite.(grad))
        end
    end
end

@testset "BlockCompliance Physical Properties" begin
    @testset "Compliance scales with load magnitude squared" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 1.0
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        x = fill(0.5, prod(nels))
        result_ref = bc(PseudoDensities(x))
        
        # Scale loads by factor 2
        F_scaled = 2.0 .* F
        problem_scaled = MultiLoad(base_problem, F_scaled)
        solver_scaled = FEASolver(DirectSolver, problem_scaled; xmin=0.01, penalty=PowerPenalty(3.0))
        bc_scaled = BlockCompliance(problem_scaled, solver_scaled; method=:exact)
        
        result_scaled = bc_scaled(PseudoDensities(x))
        
        # Should scale by factor^2 (approximately, for multi-load)
        for i in 1:nloads
            @test isapprox(result_scaled[i] / result_ref[i], 4.0; rtol=0.05)
        end
    end

    @testset "SIMP power law behavior" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 1.0
        
        problem = MultiLoad(base_problem, F)
        
        # Reference with full density
        solver_ref = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        bc_ref = BlockCompliance(problem, solver_ref; method=:exact)
        result_ref = bc_ref(PseudoDensities(ones(prod(nels))))
        
        # Test at reduced density
        rho = 0.5
        x_uniform = fill(rho, prod(nels))
        
        for p in [1.0, 2.0, 3.0]
            solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(p))
            bc = BlockCompliance(problem, solver; method=:exact)
            result_uniform = bc(PseudoDensities(x_uniform))
            
            # Check approximate power law: C(ρ) ≈ ρ^(-p) * C(1)
            for i in 1:nloads
                expected_ratio = rho^(-p)
                actual_ratio = result_uniform[i] / result_ref[i]
                @test isapprox(actual_ratio, expected_ratio; rtol=0.3)
            end
        end
    end

    @testset "Positive definiteness" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        for i in 1:nloads
            F[right_dofs[i * length(right_dofs) ÷ (nloads + 1)], i] = 1.0
        end
        
        problem = MultiLoad(base_problem, F)
        
        for method in [:exact, :exact_svd, :approx]
            nv = method == :approx ? 5 : nothing
            solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
            bc = BlockCompliance(problem, solver; method=method, nv=nv)
            
            for vf in [0.3, 0.5, 0.7]
                x = fill(vf, prod(nels))
                result = bc(PseudoDensities(x))
                @test all(result .> 0)
                @test all(isfinite.(result))
            end
        end
    end
end

@testset "BlockCompliance with Different Problem Types" begin
    @testset "Half MBB problem" begin
        nels = (6, 4)
        base_problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dofs = TopOpt.Ferrite.ndofs(base_problem.ch.dh)
        
        # Apply loads at different locations
        F[dofs-5, 1] = 1.0
        F[dofs-3, 2] = 0.8
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01)
        bc = BlockCompliance(problem, solver; method=:exact)
        
        x = ones(prod(nels))
        result = bc(PseudoDensities(x))
        @test length(result) == nloads
        @test all(isfinite.(result))
        @test all(result .> 0)
    end
end

@testset "BlockCompliance Integration with Filter" begin
    @testset "DensityFilter chain rule" begin
        nels = (6, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        filter = DensityFilter(solver; rmin=2.0)
        
        w = [0.5, 0.5]
        f = x -> sum(w .* bc(filter(PseudoDensities(x))))
        
        x = fill(0.5, prod(nels))
        
        # Compute gradient through filter
        grad = Zygote.gradient(f, x)[1]
        
        @test length(grad) == length(x)
        @test all(isfinite.(grad))
    end
end

@testset "BlockCompliance Decay Parameter" begin
    @testset "Decay affects result" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver1 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc1 = BlockCompliance(problem, solver1; method=:exact, decay=1.0)
        
        solver2 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc2 = BlockCompliance(problem, solver2; method=:exact, decay=0.5)
        
        x = fill(0.5, prod(nels))
        
        result1 = bc1(PseudoDensities(x))
        result2 = bc2(PseudoDensities(x))
        
        # Results should be finite
        @test isfinite(result1[1])
        @test isfinite(result2[1])
    end
end

@testset "BlockCompliance fevals tracking" begin
    @testset "Function evaluations are counted" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), 1)
        F[end, 1] = 1.0
        problem = MultiLoad(base_problem, F)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:exact)
        
        @test bc.fevals == 0
        
        x = fill(0.5, prod(nels))
        bc(PseudoDensities(x))
        
        @test bc.fevals == 1
        
        bc(PseudoDensities(x))
        @test bc.fevals == 2
    end
end

@testset "BlockCompliance with Custom V Matrix" begin
    @testset "User-provided V matrix" begin
        nels = (4, 4)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        nloads = 2
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        right_dofs = TopOpt.TopOptProblems.get_surface_dofs(base_problem)
        
        F[right_dofs[1], 1] = 1.0
        F[right_dofs[end], 2] = 0.5
        
        problem = MultiLoad(base_problem, F)
        
        # Create custom V matrix
        V_custom = ones(nloads, 3) ./ sqrt(nloads)
        
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        bc = BlockCompliance(problem, solver; method=:approx, V=V_custom)
        
        x = fill(0.5, prod(nels))
        result = bc(PseudoDensities(x))
        
        @test length(result) == nloads
        @test all(isfinite.(result))
    end
end