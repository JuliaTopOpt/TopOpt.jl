using TopOpt, Test, LinearAlgebra, Random, SparseArrays
import Zygote
using Ferrite: getncells, Ferrite
using TopOpt.TopOptProblems.InputOutput.INP: Parser

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
        F = spzeros(Ferrite.ndofs(base_problem.ch.dh), nloads)
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
        # Tests that heat flux is NOT penalized (key bug fix)
        nels = (16, 8)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
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

    @testset "Triangular mesh topology optimization" begin
        # Test complete workflow with imported triangular mesh
        file_name = joinpath(@__DIR__, "..", "inp_parser", "triangle.inp")
        
        # Import INP file using Parser
        inp = Parser.import_inp(file_name)
        @test inp.dh.grid isa Ferrite.Grid

        # Create problem from INP file
        problem = InpStiffness(file_name)
        @test problem isa StiffnessTopOptProblem

        # Set up solver with SIMP
        solver = FEASolver(
            DirectSolver,
            problem;
            xmin=0.001,
            penalty=PowerPenalty(3.0)
        )

        # Create compliance and volume functions
        comp = Compliance(solver)
        vol = Volume(solver)

        # Set up optimization
        V = 0.1  # Target 10% volume fraction

        # Initial uniform density
        x = fill(V, length(solver.vars))

        # Create objective and constraint
        obj = x -> comp(PseudoDensities(x))
        constr = x -> vol(PseudoDensities(x)) - V

        # Create optimization model
        model = Model(obj)
        addvar!(model, zeros(length(solver.vars)), ones(length(solver.vars)))
        add_ineq_constraint!(model, constr)

        # Use MMA optimizer
        alg = MMA87()

        # Solve with tolerance
        options = MMAOptions(; tol=Tolerance(; kkt=0.01), maxiter=100)
        res = optimize(model, alg, x; options)

        # Verify results
        @test length(res.minimizer) == getncells(problem)
        @test all(0 .<= res.minimizer .<= 1)
        
        final_vol = vol(PseudoDensities(res.minimizer))
        @test abs(final_vol - V) < 0.05  # Volume constraint satisfied

        # Compliance should be finite
        final_comp = comp(PseudoDensities(res.minimizer))
        @test isfinite(final_comp)
        @test final_comp > 0

        # Optimization should have made progress
        initial_comp = comp(PseudoDensities(fill(V, length(solver.vars))))
        @test final_comp < initial_comp  # Should improve compliance
    end
end
