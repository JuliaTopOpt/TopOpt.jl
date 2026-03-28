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

    @testset "GESO show methods" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
        
        @testset "GESO algorithm show" begin
            io = IOBuffer()
            show(io, MIME("text/plain"), geso)
            output = String(take!(io))
            @test occursin("GESO", output) || output != ""
        end

        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; seed=999)

        @testset "GESOResult show" begin
            io = IOBuffer()
            show(io, MIME("text/plain"), result)
            output = String(take!(io))
            @test occursin("GESOResult", output) || output != ""
        end
    end

    @testset "GESO with black elements (fixed solid)" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        nel = getncells(problem)
        black = falses(nel)
        black[1:5] .= true

        geso = GESO(comp, vol, 0.3, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; black=black, seed=123)

        @test all(result.topology[black] .== 1)
        @test all(x -> x == 0 || x == 1, result.topology)
        @test result isa TopOpt.Algorithms.GESOResult
        @test length(result.topology) == nel
    end

    @testset "GESO with white elements (fixed void)" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        nel = getncells(problem)
        white = falses(nel)
        white[end-4:end] .= true

        geso = GESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; white=white, seed=456)

        @test all(result.topology[white] .== 0)
        @test all(x -> x == 0 || x == 1, result.topology)
        @test result isa TopOpt.Algorithms.GESOResult
        @test length(result.topology) == nel
    end

    @testset "GESO with mixed black and white elements" begin
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        nel = getncells(problem)
        black = falses(nel)
        black[1:10] .= true
        white = falses(nel)
        white[end-9:end] .= true

        @test !any(black .& white)

        geso = GESO(comp, vol, 0.4, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; black=black, white=white, seed=789)

        @test all(result.topology[black] .== 1)
        @test all(result.topology[white] .== 0)
        @test all(x -> x == 0 || x == 1, result.topology)

        free = .!(black .| white)
        @test any(result.topology[free] .== 0)
        @test any(result.topology[free] .== 1)
    end

    @testset "GESO black/white element validation" begin
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        nel = getncells(problem)
        geso = GESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))

        black_wrong = falses(nel + 5)
        @test_throws AssertionError geso(x0; black=black_wrong, seed=100)

        white_wrong = falses(nel + 5)
        @test_throws AssertionError geso(x0; white=white_wrong, seed=101)

        black_overlap = falses(nel)
        white_overlap = falses(nel)
        black_overlap[1:3] .= true
        white_overlap[1:3] .= true
        @test_throws AssertionError geso(x0; black=black_overlap, white=white_overlap, seed=102)
    end

    @testset "GESO with black elements - different volume fractions" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        nel = getncells(problem)
        black = falses(nel)
        black[1:5] .= true

        for V_target in [0.3, 0.5, 0.7]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            geso = GESO(comp, vol, V_target, filter; maxiter=10, tol=0.1, p=1.0)
            x0 = fill(V_target, length(solver.vars))
            result = geso(x0; black=black, seed=200 + Int(100 * V_target))

            @test all(result.topology[black] .== 1)
            @test all(x -> x == 0 || x == 1, result.topology)
        end
    end

    @testset "GESO with black elements on LBeam" begin
        problem = LBeam(Val{:Linear}, Float64; force=force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        nel = getncells(problem)
        black = falses(nel)
        black[1:min(10, nel)] .= true

        geso = GESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = geso(x0; black=black, seed=300)

        @test all(result.topology[black] .== 1)
        @test all(x -> x == 0 || x == 1, result.topology)
    end
end
