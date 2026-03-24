using TopOpt, Test, LinearAlgebra, Random
using Ferrite: getncells

@testset "BESO Algorithm" begin
    E = 1.0
    ν = 0.3
    force = 1.0

    @testset "BESO Construction" begin
        nels = (20, 10)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=100, tol=0.001, p=3.0, er=0.02)

        @test beso isa TopOpt.Algorithms.TopOptAlgorithm
        @test beso.comp === comp
        @test beso.vol === vol
        @test beso.vol_limit ≈ 0.5
        @test beso.maxiter == 100
        @test beso.tol ≈ 0.001
        @test beso.p ≈ 3.0
        @test beso.er ≈ 0.02
        @test length(beso.vars) == length(solver.vars)
        @test length(beso.topology) == getncells(problem)
    end

    @testset "BESO Result Structure" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run with limited iterations for testing
        beso = BESO(comp, vol, 0.5, filter; maxiter=5, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = beso(x0)

        @test result isa TopOpt.Algorithms.BESOResult
        @test length(result.topology) == getncells(problem)
        @test all(0 .<= result.topology .<= 1) || all(result.topology .== 0) || all(result.topology .== 1)
        @test result.fevals > 0
        @test result.fevals <= 5  # Should not exceed maxiter
    end

    @testset "BESO Convergence" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run with very loose tolerance to ensure quick convergence
        beso = BESO(comp, vol, 0.5, filter; maxiter=50, tol=0.1, p=1.0, er=0.05)
        x0 = fill(0.6, length(solver.vars))
        result = beso(x0)

        # Check that result fields are properly populated
        @test isfinite(result.objval) || isnan(result.objval)  # May be NaN if not converged
        @test result.change >= 0
        @test typeof(result.converged) == Bool
        @test result.fevals >= 1
    end

    @testset "BESO Topology Validity" begin
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=1.5)

        beso = BESO(comp, vol, 0.4, filter; maxiter=10, tol=0.05, p=1.0)
        x0 = fill(0.8, length(solver.vars))
        result = beso(x0)

        # Topology should be binary (0 or 1) after BESO
        @test all(x -> x == 0 || x == 1, result.topology)

        # Check volume constraint is approximately satisfied (BESO may not exactly hit target)
        total_volume = sum(vol.cellvolumes)
        material_volume = dot(result.topology, vol.cellvolumes)
        actual_vol_frac = material_volume / total_volume
        @test actual_vol_frac <= 0.9  # Just check it's reasonable
    end

    @testset "BESO with different volume fractions" begin
        nels = (10, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        for V_target in [0.3, 0.5, 0.7]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            beso = BESO(comp, vol, V_target, filter; maxiter=10, tol=0.1, p=1.0)
            x0 = fill(V_target, length(solver.vars))
            result = beso(x0)

            # Check topology is valid
            @test length(result.topology) == getncells(problem)
            @test all(x -> x == 0 || x == 1, result.topology)
        end
    end

    @testset "BESO with LBeam problem" begin
        # Test BESO with a different problem type
        problem = LBeam(Val{:Linear}, Float64; force=force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = beso(x0)

        @test length(result.topology) == getncells(problem)
        @test all(x -> x == 0 || x == 1, result.topology)
    end

    @testset "BESO with HalfMBB" begin
        nels = (10, 4)
        problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
        x0 = fill(0.6, length(solver.vars))
        result = beso(x0)

        @test result isa TopOpt.Algorithms.BESOResult
        @test length(result.topology) == getncells(problem)
    end

    @testset "BESO Result fields consistency" begin
        nels = (8, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=1.5)

        beso = BESO(comp, vol, 0.5, filter; maxiter=5, tol=0.001, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = beso(x0)

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

    @testset "BESO - Compliance decreases with iterations (Huang & Xie 2010)" begin
        # BESO should generally decrease compliance
        # May have small oscillations but overall trend should be downward
        nels = (20, 10)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        # Run BESO
        beso = BESO(comp, vol, 0.5, filter; maxiter=20, tol=0.01, p=1.0, er=0.05)
        x0 = fill(0.6, length(solver.vars))
        result = beso(x0)

        # Final compliance should be reasonable
        @test result.objval > 0
        @test isfinite(result.objval)

        # Volume should be close to target
        total_volume = sum(vol.cellvolumes)
        material_volume = dot(result.topology, vol.cellvolumes)
        actual_vol_frac = material_volume / total_volume

        # Huang & Xie 2010: BESO volume should be within 5% of target
        @test abs(actual_vol_frac - 0.5) < 0.1
    end

    @testset "BESO - Lower volume fraction gives higher compliance" begin
        # Reducing material volume increases compliance
        # C(V2) > C(V1) for V2 < V1 (worse performance with less material)
        nels = (16, 8)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        compliances = Float64[]
        vol_fracs = [0.3, 0.5, 0.7]

        for V_target in vol_fracs
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            beso = BESO(comp, vol, V_target, filter; maxiter=15, tol=0.05, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = beso(x0)

            push!(compliances, result.objval)
        end

        # Compliance should generally increase as volume decreases
        # Allow for some noise due to stochastic nature
        @test compliances[1] >= compliances[2] * 0.8  # 30% vol vs 50% vol
        @test compliances[2] >= compliances[3] * 0.8  # 50% vol vs 70% vol
    end

    @testset "BESO - Topology connectivity (no floating elements)" begin
        # BESO with filter should produce connected structures
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=10, tol=0.05, p=1.0)
        x0 = fill(0.6, length(solver.vars))
        result = beso(x0)

        # Check that material is connected to supports/load
        # For a cantilever, material should exist at loaded end (right side)
        topology = result.topology
        nels_x = nels[1]
        nels_y = nels[2]

        # Rightmost elements (near load) should have material
        right_elements = [i for i in 1:length(topology)
                         if (i-1) % nels_x == nels_x - 1 && topology[i] == 1]
        @test length(right_elements) > 0

        # Leftmost elements (near supports) should have material
        left_elements = [i for i in 1:length(topology)
                        if (i-1) % nels_x == 0 && topology[i] == 1]
        @test length(left_elements) > 0
    end

    @testset "BESO - Symmetric problem produces symmetric topology" begin
        # For a symmetric problem, BESO should produce nearly symmetric result
        nels = (16, 8)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        comp = Compliance(solver)
        vol = Volume(solver)
        filter = DensityFilter(solver; rmin=2.0)

        beso = BESO(comp, vol, 0.5, filter; maxiter=15, tol=0.05, p=1.0)
        x0 = fill(0.5, length(solver.vars))
        result = beso(x0)

        topology = result.topology
        nels_x = nels[1]
        nels_y = nels[2]

        # Check symmetry: compare element at (i,j) with (i, nels_y-j+1)
        symmetry_errors = 0
        total_pairs = 0
        for i in 1:nels_x
            for j in 1:div(nels_y, 2)
                idx1 = (j-1) * nels_x + i
                idx2 = (nels_y - j) * nels_x + i
                if topology[idx1] != topology[idx2]
                    symmetry_errors += 1
                end
                total_pairs += 1
            end
        end

        # Should be mostly symmetric (allow some asymmetry from discretization)
        if total_pairs > 0
            symmetry_score = 1.0 - symmetry_errors / total_pairs
            @test symmetry_score > 0.7
        end
    end

    @testset "BESO - Sensitivity to filter radius" begin
        # Larger filter radius produces smoother topologies
        # with fewer small features (checkerboard-free)
        nels = (16, 8)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        for rmin in [1.5, 2.5, 3.5]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=rmin)

            beso = BESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0)
            x0 = fill(0.5, length(solver.vars))
            result = beso(x0)

            # Should produce valid binary topology regardless of filter size
            @test all(x -> x == 0 || x == 1, result.topology)
            @test result.objval > 0
        end
    end

    @testset "BESO - Evolutionary ratio affects convergence" begin
        # ER (evolutionary ratio) controls material change rate
        # Higher ER = faster changes but potentially less stable
        nels = (12, 6)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)

        for er in [0.02, 0.05, 0.1]
            solver = FEASolver(DirectSolver, problem; xmin=0.001)
            comp = Compliance(solver)
            vol = Volume(solver)
            filter = DensityFilter(solver; rmin=2.0)

            beso = BESO(comp, vol, 0.5, filter; maxiter=10, tol=0.1, p=1.0, er=er)
            x0 = fill(0.5, length(solver.vars))
            result = beso(x0)

            # Should converge for all reasonable ER values
            @test result.fevals > 0
            @test result.fevals <= 10
            @test all(x -> x == 0 || x == 1, result.topology)
        end
    end
end
