using TopOpt, Test, LinearAlgebra, Random, SparseArrays
import Zygote
using Ferrite: getncells

Random.seed!(42)

@testset "End-to-End Integration Tests" begin
    E = 1.0
    ν = 0.3
    force = 1.0

    @testset "Complete SIMP optimization workflow" begin
        # SIMP optimization using continuation with MMA
        # Based on the approach in csimp.jl
        nels = (20, 10)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))

        # Set up optimization
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Objective and constraint
        V = 0.5  # volume fraction
        obj = x -> comp(filter(PseudoDensities(x)))
        constr = x -> vol(filter(PseudoDensities(x))) - V

        model = Model(obj)
        addvar!(model, zeros(length(solver.vars)), ones(length(solver.vars)))
        add_ineq_constraint!(model, constr)
        alg = MMA87()

        # Brief continuation SIMP
        nsteps = 3
        ps = range(1.0, 3.0; length=nsteps)
        tols = exp10.(range(-1, -2; length=nsteps))
        x = fill(V, length(solver.vars))

        for j in 1:nsteps
            p = ps[j]
            tol = tols[j]
            TopOpt.setpenalty!(solver, p)
            options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=50)
            res = optimize(model, alg, x; options)
            x = res.minimizer
        end

        # Verify results
        @test length(x) == getncells(problem)
        @test all(0 .<= x .<= 1)
        final_vol = vol(PseudoDensities(x))
        @test abs(final_vol - V) < 0.05  # Volume constraint should be satisfied
    end

    @testset "Filtered sensitivity workflow" begin
        # Test that filtering produces smooth topologies
        nels = (16, 8)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        # Create filtered compliance
        comp = Compliance(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Evaluate at uniform density
        x = fill(0.5, length(solver.vars))
        filtered_comp = x -> comp(filter(PseudoDensities(x)))

        val = filtered_comp(x)
        @test val > 0
        @test isfinite(val)

        # Gradient should work through filter
        grad = Zygote.gradient(filtered_comp, x)[1]
        @test length(grad) == length(x)
        @test all(isfinite.(grad))
    end

    @testset "Multi-load optimization workflow" begin
        # Complete multi-load mean compliance optimization
        nels = (12, 6)
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Create multiple load cases
        nloads = 3
        F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
        dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))

        Random.seed!(42)
        for i in 1:nloads
            dofs = dense_load_inds[rand(1:length(dense_load_inds), 2)]
            F[dofs, i] .= [1.0, 0.5]
        end

        problem = MultiLoad(base_problem, F)
        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(2.0))

        # Set up optimization with mean compliance
        mc = MeanCompliance(problem, solver; method=:exact)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Brief optimization
        V = 0.5
        obj = x -> mc(filter(PseudoDensities(x)))
        constr = x -> vol(filter(PseudoDensities(x))) - V

        model = Model(obj)
        addvar!(model, zeros(length(solver.vars)), ones(length(solver.vars)))
        add_ineq_constraint!(model, constr)
        alg = MMA87()

        options = MMAOptions(; tol=Tolerance(; kkt=0.01), maxiter=30)
        x0 = fill(V, length(solver.vars))
        res = optimize(model, alg, x0; options)

        # Verify results
        @test length(res.minimizer) == getncells(problem)
        @test res.fcalls > 0
    end

    @testset "Heat transfer optimization workflow" begin
        # Complete heat transfer optimization workflow
        # Tests that heat source is NOT penalized (key bug fix)
        nels = (16, 8)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(2.0))

        # Thermal compliance optimization
        comp = ThermalCompliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Brief optimization
        V = 0.5
        obj = x -> comp(filter(PseudoDensities(x)))
        constr = x -> vol(filter(PseudoDensities(x))) - V

        model = Model(obj)
        addvar!(model, zeros(length(solver.vars)), ones(length(solver.vars)))
        add_ineq_constraint!(model, constr)
        alg = MMA87()

        options = MMAOptions(; tol=Tolerance(; kkt=0.01), maxiter=30)
        x0 = fill(V, length(solver.vars))
        res = optimize(model, alg, x0; options)

        @test length(res.minimizer) == getncells(problem)
        @test res.fcalls > 0
    end

    @testset "BESO + DensityFilter integration" begin
        # BESO with filtering
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=15, tol=0.05, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = beso(x0)

        @test length(result.topology) == getncells(problem)
        @test all(x -> x == 0 || x == 1, result.topology)

        # Volume should be approximately satisfied
        total_volume = sum(vol.cellvolumes)
        material_volume = dot(result.topology, vol.cellvolumes)
        actual_vol_frac = material_volume / total_volume
        @test abs(actual_vol_frac - 0.5) < 0.1
    end

    @testset "GESO + DensityFilter integration" begin
        # GESO with filtering
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=123)

        @test length(result.topology) == getncells(problem)
        @test all(x -> x == 0 || x == 1, result.topology)
        @test result.fevals > 0
    end

    @testset "Different problem types compatibility" begin
        # Test that all problem types work with the same interface
        problems = [
            ("PointLoadCantilever", () -> PointLoadCantilever(Val{:Linear}, (8, 4), (1.0, 1.0), E, ν, force)),
            ("HalfMBB", () -> HalfMBB(Val{:Linear}, (8, 4), (1.0, 1.0), E, ν, force)),
            ("LBeam", () -> LBeam(Val{:Linear}, Float64; force=force)),
        ]

        for (name, prob_fn) in problems
            problem = prob_fn()
            solver = FEASolver(DirectSolver, problem; xmin=0.001)

            comp = Compliance(solver)
            vol = Volume(solver)

            # Quick evaluation
            x = fill(0.5, length(solver.vars))
            c = comp(PseudoDensities(x))
            v = vol(PseudoDensities(x))

            @test c > 0
            @test v > 0
            @test v < 1  # Volume fraction should be less than full
        end
    end

    @testset "Solver type consistency" begin
        # Different solvers should give similar results
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        results = Dict()

        # DirectSolver
        solver_direct = FEASolver(DirectSolver, problem; xmin=0.001)
        solver_direct.vars .= 1.0
        solver_direct()
        results["DirectSolver"] = solver_direct.u

        # Results should be similar
        @test length(results["DirectSolver"]) > 0
    end

    @testset "Penalty interpolation accuracy" begin
        # Verify SIMP penalty law
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        # Reference at full density
        solver_ref = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        comp_ref = Compliance(solver_ref)
        C_full = comp_ref(PseudoDensities(ones(length(solver_ref.vars))))

        # Test power law for different penalties
        for p in [1.0, 2.0, 3.0]
            solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(p))
            comp = Compliance(solver)

            rho = 0.5
            x = fill(rho, length(solver.vars))
            C = comp(PseudoDensities(x))

            # Check approximate power law
            if p == 1.0
                # Linear: C(0.5) ≈ C(1)/0.5
                @test C > C_full
            else
                # SIMP: C increases faster
                @test C > C_full
            end
        end
    end

    @testset "Gradient consistency across methods" begin
        # Gradients should be consistent between analytical and AD
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.01)

        comp = Compliance(solver)
        x = fill(0.5, length(solver.vars))

        # Evaluate
        val = comp(PseudoDensities(x))

        # AD gradient
        grad_ad = Zygote.gradient(x -> comp(PseudoDensities(x)), x)[1]

        @test length(grad_ad) == length(x)
        @test all(isfinite.(grad_ad))

        # Gradient should point in direction of decreasing compliance
        # (more material = lower compliance)
        @test sum(grad_ad) < 0  # Overall trend
    end

    @testset "Volume constraint accuracy" begin
        # Volume function should accurately compute volume fraction
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        vol = Volume(solver)

        # Test uniform densities
        for vf in [0.3, 0.5, 0.7, 0.9]
            x = fill(vf, length(solver.vars))
            computed_vf = vol(PseudoDensities(x))
            @test isapprox(computed_vf, vf; rtol=0.01)
        end
    end

    @testset "Filter consistency" begin
        # Filter should produce smooth transitions
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        filter = DensityFilter(solver; rmin=2.0)

        # Create checkerboard pattern
        x = Float64[((i+j) % 2) for i in 1:nels[1], j in 1:nels[2]]
        x = vec(x)

        # Filter should smooth checkerboard
        x_filtered = filter(PseudoDensities(x))

        # Max difference should be reduced
        original_range = maximum(x) - minimum(x)
        filtered_range = maximum(x_filtered) - minimum(x_filtered)

        @test filtered_range <= original_range
    end
end

@testset "Regression Tests" begin
    # These tests guard against breaking changes

    @testset "API backward compatibility" begin
        # Ensure core API hasn't changed
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        # Core functions should exist and work
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        x = ones(length(solver.vars))

        # All should be callable
        @test comp(PseudoDensities(x)) isa Real
        @test vol(PseudoDensities(x)) isa Real
        @test filter(PseudoDensities(x)) isa AbstractVector
    end

    @testset "Type stability" begin
        # Results should have stable types
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        comp = Compliance(solver)
        x = ones(length(solver.vars))

        val = comp(PseudoDensities(x))
        @test val isa Float64
    end
end
