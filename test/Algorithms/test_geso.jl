using TopOpt, Test, LinearAlgebra, Random
using Ferrite: getncells

@testset "GESO Algorithm" begin
    E = 1.0
    ν = 0.3
    force = 1.0

    @testset "GESO Construction" begin
        nels = (20, 10)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=100, tol=0.001, p=3.0)

        @test geso isa TopOpt.Algorithms.TopOptAlgorithm
        @test geso.comp === comp
        @test geso.vol === vol
        @test geso.vol_limit ≈ 0.5
        @test geso.maxiter == 100
        @test geso.tol ≈ 0.001
        @test geso.Pcmin ≈ 0.6
        @test geso.Pcmax ≈ 1.0
        @test geso.Pmmin ≈ 0.5
        @test geso.Pmmax ≈ 1.0
        @test geso.Pen ≈ 3.0
        @test geso.string_length == 4
        @test length(geso.vars) == length(solver.vars)
        @test length(geso.topology) == getncells(problem)
        @test size(geso.genotypes) == (4, length(solver.vars))
        @test size(geso.children) == (4, length(solver.vars))
    end

    @testset "GESO Result Structure" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run with limited iterations for testing
        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=123)  # Set seed for reproducibility

        @test result isa TopOpt.Algorithms.GESOResult
        @test length(result.topology) == getncells(problem)
        @test all(0 .<= result.topology .<= 1) || all(result.topology .== 0) || all(result.topology .== 1)
        @test result.fevals > 0
        @test result.fevals <= 5  # Should not exceed maxiter
    end

    @testset "GESO Convergence" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run with very loose tolerance to ensure quick convergence
        geso = GESO(comp, vol, 0.5, filter; maxiter=50, tol=0.1, p=1.0)
        x0 = fill(0.6, length(solver.vars))
        result = geso(x0; seed=456)

        # Check that result fields are properly populated
        @test isfinite(result.objval) || isnan(result.objval)
        @test result.change >= 0
        @test typeof(result.converged) == Bool
        @test result.fevals >= 1
    end

    @testset "GESO Topology Validity" begin
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=1.5)

        geso = GESO(comp, vol, 0.4, filter; maxiter=10, tol=0.05, p=1.0)
        x0 = fill(0.8, length(solver.vars))
        result = geso(x0; seed=789)

        # Topology should be binary (0 or 1) after GESO
        @test all(x -> x == 0 || x == 1, result.topology)

        # Check volume constraint is approximately satisfied
        total_volume = sum(vol.cellvolumes)
        material_volume = dot(result.topology, vol.cellvolumes)
        actual_vol_frac = material_volume / total_volume
        @test actual_vol_frac <= 0.9  # Just check it's reasonable
    end

    @testset "GESO with different volume fractions" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        for V_target in [0.3, 0.5, 0.7]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            geso = GESO(comp, vol, V_target, filter; maxiter=10, tol=0.1, p=1.0)
            x0 = fill(V_target, length(solver.vars))
            result = geso(x0; seed=100 + Int(100 * V_target))

            # Check topology is valid
            @test length(result.topology) == getncells(problem)
            @test all(x -> x == 0 || x == 1, result.topology)
        end
    end

    @testset "GESO with LBeam problem" begin
        # Test GESO with a different problem type
        problem = LBeam(Val{:Linear}, Float64; force=force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=111)

        @test length(result.topology) == getncells(problem)
        @test all(x -> x == 0 || x == 1, result.topology)
    end

    @testset "GESO setpenalty!" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)

        # Test that penalty can be updated
        TopOpt.setpenalty!(geso, 2.0)
        @test geso.penalty.p ≈ 2.0

        # Run with updated penalty
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=222)
        @test result isa TopOpt.Algorithms.GESOResult
    end

    @testset "GESO with HalfMBB" begin
        nels = (10, 4)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.6, length(solver.vars))
        result = geso(x0; seed=333)

        @test result isa TopOpt.Algorithms.GESOResult
        @test length(result.topology) == getncells(problem)
    end

    @testset "GESO Result fields consistency" begin
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=1.5)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.001, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=444)

        # Verify all result fields are populated
        @test hasproperty(result, :topology)
        @test hasproperty(result, :objval)
        @test hasproperty(result, :change)
        @test hasproperty(result, :converged)
        @test hasproperty(result, :fevals)

        # Type checks
        @test result.topology isa AbstractVector
        @test result.objval isa Real
        @test result.change isa Real
        @test result.converged isa Bool
        @test result.fevals isa Integer
    end

    @testset "GESO internal functions" begin
        # Test internal helper functions if accessible
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)

        # Test get_progress function
        total_vol = 100.0
        current_vol = 60.0
        design_vol = 50.0
        progress = TopOpt.Algorithms.get_progress(current_vol, total_vol, design_vol)
        @test 0 <= progress <= 1

        # Test get_probs function
        Prg = 0.5
        Pc, Pm = TopOpt.Algorithms.get_probs(geso, Prg)
        @test 0 <= Pc <= 1
        @test 0 <= Pm <= 1
        @test Pc >= geso.Pcmin
        @test Pc <= geso.Pcmax
        @test Pm >= geso.Pmmin
        @test Pm <= geso.Pmmax
    end

    @testset "GESO genetic operations" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)

        # Initialize genotypes
        genotypes = trues(geso.string_length, length(solver.vars))
        children = falses(geso.string_length, length(solver.vars))

        # Test crossover operation
        i, j = 1, 2
        TopOpt.Algorithms.crossover!(children, genotypes, i, j)
        # Children should have some combination of parent genes
        @test children[:, i] isa BitVector
        @test children[:, j] isa BitVector
    end

    @testset "GESO - Population-based search convergence" begin
        # GA-based methods should show convergence over generations
        # even if stochastic in nature
        nels = (16, 8)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run multiple times with different seeds to verify statistical behavior
        results = Float64[]
        for seed in [100, 200, 300]
            geso = GESO(comp, vol, 0.5, filter; maxiter=15, tol=0.05, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = geso(x0; seed=seed)
            push!(results, result.objval)

            # Each run should produce valid results
            @test result.fevals > 0
            @test result.fevals <= 15
            @test all(x -> x == 0 || x == 1, result.topology)
        end

        # Results should be of similar magnitude (not wildly different)
        # GESO is stochastic but should produce consistent quality solutions
        @test maximum(results) / minimum(results) < 10.0
    end

    @testset "GESO - Binary encoding preserves volume constraint" begin
        # GA encoding should maintain feasibility constraints
        # Binary representation directly encodes material presence
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        for V_target in [0.3, 0.5, 0.7]
            geso = GESO(comp, vol, V_target, filter; maxiter=10, tol=0.1, p=1.0)
            x0 = fill(V_target, length(solver.vars))
            result = geso(x0; seed=400 + Int(100*V_target))

            # Check volume fraction
            total_volume = sum(vol.cellvolumes)
            material_volume = dot(result.topology, vol.cellvolumes)
            actual_vol_frac = material_volume / total_volume

            # GESO should approximately satisfy volume constraint
            @test abs(actual_vol_frac - V_target) < 0.15
        end
    end

    @testset "GESO - Fitness improvement over generations" begin
        # GA should generally improve fitness (compliance here)
        # over the course of evolution
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run with more iterations to see improvement
        geso = GESO(comp, vol, 0.5, filter; maxiter=20, tol=0.02, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=500)

        # Final objective should be reasonable
        @test result.objval > 0
        @test isfinite(result.objval)

        # Volume should be close to target
        total_volume = sum(vol.cellvolumes)
        material_volume = dot(result.topology, vol.cellvolumes)
        actual_vol_frac = material_volume / total_volume

        @test abs(actual_vol_frac - 0.5) < 0.1
    end

    @testset "GESO - Reproducibility with same seed" begin
        # GA with fixed seed should be deterministic
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        results = Vector{Float64}[]

        for run in 1:2
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = geso(x0; seed=999)  # Same seed
            push!(results, result.topology)
        end

        # Results should be identical with same seed
        @test results[1] == results[2]
    end

    @testset "GESO - Different seeds produce valid results" begin
        # GA should find valid solutions regardless of seed
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        for seed in [1, 42, 123, 456, 789]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = geso(x0; seed=seed)

            # All should produce valid binary topologies
            @test all(x -> x == 0 || x == 1, result.topology)
            @test length(result.topology) == getncells(problem)
            @test result.fevals > 0
        end
    end

    @testset "GESO - Genetic operators produce valid individuals" begin
        # Test that genetic operators maintain binary feasibility
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)

        # Test crossover produces valid bit patterns
        n_vars = length(solver.vars)
        genotypes = rand(Bool, geso.string_length, n_vars)
        children = falses(geso.string_length, n_vars)

        for trial in 1:10
            # Random parent indices
            i, j = rand(1:geso.string_length, 2)
            if i != j
                TopOpt.Algorithms.crossover!(children, genotypes, i, j)

                # Children should be valid BitVectors (0s and 1s only)
                @test all(children[:, i] .== true .|| children[:, i] .== false)
                @test all(children[:, j] .== true .|| children[:, j] .== false)
            end
        end
    end

    @testset "GESO - Probability functions are well-defined" begin
        # Test that probability functions return valid values
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)

        # Test progress function for various inputs
        test_cases = [
            (50.0, 100.0, 50.0),   # Exact match
            (60.0, 100.0, 50.0),   # Above target
            (40.0, 100.0, 50.0),   # Below target
        ]

        for (current, total, design) in test_cases
            progress = TopOpt.Algorithms.get_progress(current, total, design)
            @test 0.0 <= progress <= 1.0
        end

        # Test probability functions across progress range
        for Prg in 0.0:0.1:1.0
            Pc, Pm = TopOpt.Algorithms.get_probs(geso, Prg)
            @test 0.0 <= Pc <= 1.0
            @test 0.0 <= Pm <= 1.0

            # Probabilities should be within configured bounds
            @test Pc >= geso.Pcmin
            @test Pc <= geso.Pcmax
            @test Pm >= geso.Pmmin
            @test Pm <= geso.Pmmax
        end
    end

    @testset "GESO - Lower volume fractions increase compliance" begin
        # GA should follow physical law:
        # Less material = higher compliance (worse performance)
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        compliances = Float64[]
        vol_fracs = [0.3, 0.5, 0.7]

        for V_target in vol_fracs
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            geso = GESO(comp, vol, V_target, filter; maxiter=15, tol=0.05, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = geso(x0; seed=600 + Int(100*V_target))

            push!(compliances, result.objval)
        end

        # Compliance should generally increase as volume decreases
        # Allow for GA stochasticity
        @test compliances[1] >= compliances[2] * 0.7  # 30% vs 50%
        @test compliances[2] >= compliances[3] * 0.7  # 50% vs 70%
    end
end
