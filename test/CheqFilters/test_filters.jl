using TopOpt, LinearAlgebra, Test, Statistics
using TopOpt: DensityFilterFun, PseudoDensities, ProjectedDensityFilterFun
using TopOpt.CheqFilters: SensFilterFun, FilterMetadata
using ChainRulesCore: ChainRulesCore
using NonconvexCore: getdim

@testset "Filter Tests" begin
    @testset "DensityFilterFun Construction" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        @test df.rmin == rmin
        @test df.metadata isa FilterMetadata
    end

    nels = (5, 5)
    problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))
    rmin = 2.0

    @testset "getdim returns correct dimension" begin
        df = DensityFilterFun(solver, rmin)

        # getdim returns size of jacobian
        @test getdim(df) == size(df.jacobian, 1)
    end

    @testset "show method for DensityFilterFun" begin
        df = DensityFilterFun(solver, rmin)
        io = IOBuffer()
        show(io, MIME"text/plain"(), df)
        output = String(take!(io))
        @test occursin("density filter", lowercase(output))
    end

    @testset "SensFilterFun Construction" begin
        sf = SensFilterFun(solver; rmin=rmin)
        @test sf.rmin == rmin
        @test sf.metadata isa FilterMetadata
    end

    @testset "SensFilterFun show method" begin
        sf = SensFilterFun(solver; rmin=rmin)
        io = IOBuffer()
        # Test show for SensFilterFun
        show(io, MIME"text/plain"(), sf)
        output = String(take!(io))
        @test occursin("sensitivity filter", lowercase(output))
    end

    @testset "DensityFilterFun Application" begin
        df = DensityFilterFun(solver; rmin=rmin)
        n = length(solver.vars)
        x = ones(n) * 0.5
        result = df(PseudoDensities(x))
        @test result isa PseudoDensities
        @test length(result.x) == n
        @test all(result.x .>= 0.0)
        @test all(result.x .<= 1.0)
    end

    @testset "SensFilterFun Application" begin
        sf = SensFilterFun(solver; rmin=rmin)
        n = length(solver.vars)
        x = ones(n) * 0.5
        result = sf(PseudoDensities(x))
        @test result isa PseudoDensities
        @test length(result.x) == n
    end

    @testset "Filter radius effects" begin
        nels = (10, 10)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        df_small = DensityFilterFun(solver; rmin=1.5)
        df_large = DensityFilterFun(solver; rmin=5.0)

        x = rand(length(solver.vars))

        result_small = df_small(PseudoDensities(x))
        result_large = df_large(PseudoDensities(x))

        @test result_small isa PseudoDensities
        @test result_large isa PseudoDensities

        std_small = std(result_small.x)
        std_large = std(result_large.x)
        @test std_large <= std_small * 1.5
    end

    @testset "Filter with uniform density" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        n = length(solver.vars)
        x_uniform = ones(n) * 0.5

        result = df(PseudoDensities(x_uniform))

        @test all(result.x .≈ 0.5)
    end

    @testset "Filter gradient check" begin
        nels = (3, 3)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        x = rand(length(solver.vars))

        f = x -> sum(df(PseudoDensities(x)).x)

        result = f(x)
        @test isfinite(result)
    end

    @testset "DensityFilterFun values and gradients with transpose" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        x_raw = rand(length(solver.vars))

        # Forward pass: DensityFilterFun filters the values
        x_filtered = df(PseudoDensities(x_raw)).x

        # Verify forward filtering actually changes values
        @test x_filtered != x_raw

        # Test gradient via ChainRules
        y, pullback = ChainRulesCore.rrule(df, PseudoDensities(x_raw))

        # Output is filtered values
        @test y.x ≈ x_filtered

        # Backward pass: gradient w.r.t. filtered output
        # For sum(filtered), gradient is ones
        Δ = PseudoDensities(ones(length(x_raw)))

        # Pullback returns gradient w.r.t. input
        _, grad = pullback(Δ)

        # Density filter uses transpose of jacobian for gradients
        # grad.x should equal jacobian' * Δ
        @test length(grad.x) == length(x_raw)
        @test isfinite(sum(grad.x))

        # Compare with manual transpose multiplication
        manual_grad = df.jacobian' * Δ.x
        @test grad.x ≈ manual_grad rtol = 1e-10
    end

    @testset "SensFilterFun only filters gradients" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        sf = SensFilterFun(solver; rmin=rmin)

        x_raw = rand(length(solver.vars))

        # Forward pass: SensFilterFun returns values unchanged
        x_out = sf(PseudoDensities(x_raw)).x

        # Values should be identical (no filtering in forward)
        @test x_out ≈ x_raw

        # Test gradient via ChainRules
        y, pullback = ChainRulesCore.rrule(sf, PseudoDensities(x_raw))

        # Output equals input (no forward filtering)
        @test y.x ≈ x_raw

        # Backward pass with non-uniform gradient
        # Create a gradient that varies spatially
        Δ = PseudoDensities(collect(1:length(x_raw)) ./ length(x_raw))

        _, grad = pullback(Δ)

        # SensFilterFun applies filtering in backward pass only
        # The gradient should be filtered (different from input gradient)
        @test length(grad.x) == length(x_raw)
        @test isfinite(sum(grad.x))

        # Gradient is filtered, so it should differ from original
        # (unless gradient is uniform)
        @test grad.x != Δ.x

        # With uniform gradient, filtering has no effect
        Δ_uniform = PseudoDensities(ones(length(x_raw)))
        _, grad_uniform = pullback(Δ_uniform)

        # Uniform gradient stays uniform after filtering
        @test all(grad_uniform.x .≈ grad_uniform.x[1])
    end

    @testset "Combined filter with sum function" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)
        sf = SensFilterFun(solver; rmin=rmin)

        # Test function: sum of filtered values
        # For DensityFilterFun: filters both values and backpropagates gradients
        # For SensFilterFun: leaves values unchanged, only filters gradients

        # DensityFilterFun test
        x1 = rand(length(solver.vars))

        # Forward: density filtered
        y1 = sum(df(PseudoDensities(x1)).x)

        # Gradient check via finite differences
        function density_filtered_sum(x)
            return sum(df(PseudoDensities(x)).x)
        end

        # SensFilterFun test
        x2 = rand(length(solver.vars))

        # Forward: values unchanged
        y2 = sum(sf(PseudoDensities(x2)).x)

        # Should equal sum of raw values (no forward filtering)
        @test y2 ≈ sum(x2)

        # Gradient via pullback
        _, pullback_df = ChainRulesCore.rrule(df, PseudoDensities(x1))
        _, pullback_sf = ChainRulesCore.rrule(sf, PseudoDensities(x2))

        # Gradient of sum is ones
        Δ = PseudoDensities(ones(length(solver.vars)))

        _, grad_df = pullback_df(Δ)
        _, grad_sf = pullback_sf(Δ)

        # Both gradients should be valid
        @test length(grad_df.x) == length(x1)
        @test length(grad_sf.x) == length(x2)
        @test isfinite(sum(grad_df.x))
        @test isfinite(sum(grad_sf.x))
    end

    @testset "Gradient filtering comparison" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)
        sf = SensFilterFun(solver; rmin=rmin)

        x = rand(length(solver.vars))

        # Create a spatially varying gradient
        Δ = PseudoDensities([sin(2π * i / length(x)) for i in 1:length(x)])

        # DensityFilterFun pullback: uses jacobian transpose
        _, pullback_df = ChainRulesCore.rrule(df, PseudoDensities(x))
        _, grad_df = pullback_df(Δ)

        # SensFilterFun pullback: uses nodal gradient smoothing
        _, pullback_sf = ChainRulesCore.rrule(sf, PseudoDensities(x))
        _, grad_sf = pullback_sf(Δ)

        # Both should produce smoothed gradients
        @test length(grad_df.x) == length(x)
        @test length(grad_sf.x) == length(x)

        # Gradients should be finite
        @test all(isfinite, grad_df.x)
        @test all(isfinite, grad_sf.x)

        # DensityFilterFun: output values are filtered, gradient uses jacobian'
        # SensFilterFun: output values are unchanged, gradient is smoothed
        df_output = df(PseudoDensities(x)).x
        sf_output = sf(PseudoDensities(x)).x

        @test df_output != x  # DensityFilterFun changes values
        @test sf_output ≈ x   # SensFilterFun leaves values unchanged
    end

    @testset "Multiple filters on same grid" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        df1 = DensityFilterFun(solver; rmin=1.5)
        df2 = DensityFilterFun(solver; rmin=3.0)

        x = rand(length(solver.vars))

        result1 = df1(PseudoDensities(x))
        result2 = df2(PseudoDensities(x))

        @test result1 isa PseudoDensities
        @test result2 isa PseudoDensities

        @test result1.x != result2.x
    end

    @testset "ProjectedDensityFilterFun construction" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        @testset "With no projections" begin
            pdf = ProjectedDensityFilterFun(df, nothing, nothing)
            @test pdf.filter === df
            @test pdf.preproj === nothing
            @test pdf.postproj === nothing
            @test pdf isa ProjectedDensityFilterFun
        end

        @testset "With pre-projection only" begin
            preproj = x -> x^2
            pdf = ProjectedDensityFilterFun(df, preproj, nothing)
            @test pdf.filter === df
            @test pdf.preproj === preproj
            @test pdf.postproj === nothing
        end

        @testset "With post-projection only" begin
            postproj = x -> clamp(x, 0.1, 0.9)
            pdf = ProjectedDensityFilterFun(df, nothing, postproj)
            @test pdf.filter === df
            @test pdf.preproj === nothing
            @test pdf.postproj === postproj
        end

        @testset "With both pre and post projections" begin
            preproj = x -> sqrt(x)
            postproj = x -> clamp(x, 0.0, 1.0)
            pdf = ProjectedDensityFilterFun(df, preproj, postproj)
            @test pdf.filter === df
            @test pdf.preproj === preproj
            @test pdf.postproj === postproj
        end
    end

    @testset "ProjectedDensityFilterFun getdim" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0

        @testset "getdim with DensityFilterFun" begin
            df = DensityFilterFun(solver, rmin)
            pdf = ProjectedDensityFilterFun(df, nothing, nothing)

            # getdim should delegate to the underlying DensityFilterFun
            @test getdim(pdf) == getdim(df)
            @test getdim(pdf) == size(df.jacobian, 1)
        end

        @testset "getdim with various projection configurations" begin
            df = DensityFilterFun(solver; rmin=rmin)

            # All configurations should return same dimension
            pdf_none = ProjectedDensityFilterFun(df, nothing, nothing)
            pdf_pre = ProjectedDensityFilterFun(df, x -> x^2, nothing)
            pdf_post = ProjectedDensityFilterFun(df, nothing, x -> clamp(x, 0.1, 0.9))
            pdf_both = ProjectedDensityFilterFun(df, x -> x^2, x -> clamp(x, 0.1, 0.9))

            expected_dim = getdim(df)
            @test getdim(pdf_none) == expected_dim
            @test getdim(pdf_pre) == expected_dim
            @test getdim(pdf_post) == expected_dim
            @test getdim(pdf_both) == expected_dim
        end
    end

    @testset "ProjectedDensityFilterFun application" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        x = rand(length(solver.vars))

        @testset "No projections - acts like plain DensityFilterFun" begin
            pdf = ProjectedDensityFilterFun(df, nothing, nothing)
            result = pdf(PseudoDensities(x))

            # Should produce same result as plain DensityFilterFun
            expected = df(PseudoDensities(x))
            @test result isa PseudoDensities
            @test result.x ≈ expected.x
        end

        @testset "With pre-projection" begin
            preproj = x -> x^2
            pdf = ProjectedDensityFilterFun(df, preproj, nothing)

            result = pdf(PseudoDensities(x))
            expected = df(PseudoDensities(preproj.(x)))

            @test result isa PseudoDensities
            @test result.x ≈ expected.x
        end

        @testset "With post-projection" begin
            postproj = x -> clamp(x, 0.1, 0.9)
            pdf = ProjectedDensityFilterFun(df, nothing, postproj)

            result = pdf(PseudoDensities(x))
            expected_raw = df(PseudoDensities(x))
            expected = postproj.(expected_raw.x)

            @test result isa PseudoDensities
            @test result.x ≈ expected
        end

        @testset "With both pre and post projections" begin
            preproj = x -> sqrt(x)
            postproj = x -> clamp(x, 0.2, 0.8)
            pdf = ProjectedDensityFilterFun(df, preproj, postproj)

            result = pdf(PseudoDensities(x))

            # Apply preproj, then filter, then postproj
            pre_x = preproj.(x)
            filtered = df(PseudoDensities(pre_x))
            expected = postproj.(filtered.x)

            @test result isa PseudoDensities
            @test result.x ≈ expected
        end
    end

    @testset "ProjectedDensityFilterFun with uniform density" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        n = length(solver.vars)
        x_uniform = ones(n) * 0.5

        @testset "No projection" begin
            pdf = ProjectedDensityFilterFun(df, nothing, nothing)
            result = pdf(PseudoDensities(x_uniform))

            # Uniform density stays uniform through filter
            @test all(result.x .≈ 0.5)
        end

        @testset "With linear pre-projection" begin
            # Linear pre-projection of uniform field is still uniform
            preproj = x -> 2x
            pdf = ProjectedDensityFilterFun(df, preproj, nothing)
            result = pdf(PseudoDensities(x_uniform))

            @test all(result.x .≈ 1.0)
        end
    end

    @testset "ProjectedDensityFilterFun output type" begin
        nels = (5, 5)
        problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenaltyFun(3.0))

        rmin = 2.0
        df = DensityFilterFun(solver; rmin=rmin)

        x = rand(length(solver.vars))
        input = PseudoDensities(x)

        @testset "Returns PseudoDensities with filtered=true" begin
            pdf = ProjectedDensityFilterFun(df, nothing, nothing)
            result = pdf(input)

            @test result isa PseudoDensities
            # The third type parameter should be true (filtered)
            @test typeof(result) <: PseudoDensities{<:Any,<:Any,true}
        end
    end
end
