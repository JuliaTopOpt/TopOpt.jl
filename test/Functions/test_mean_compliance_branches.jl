using TopOpt, Test, LinearAlgebra, Random, SparseArrays
using TopOpt: Nonconvex
using Statistics: mean

# Test the conditional branches in MeanCompliance constructor
# These tests specifically target the branching logic in lines ~18-43 of mean_compliance.jl

Random.seed!(42)

@testset "MeanCompliance Constructor Branches" begin

    # Common test setup
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

    @testset "Branch 1: method=:trace creates TraceEstimationMean" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test with default parameters
        mc = MeanCompliance(problem, solver; method=:trace)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean

        # Test with nv specified
        mc = MeanCompliance(problem, solver; method=:trace, nv=5)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 5

        # Test with sample_method=:hadamard
        mc = MeanCompliance(problem, solver; method=:trace, nv=5, sample_method=:hadamard)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test mc.method.sample_once == true  # hadamard forces sample_once=true

        # Test with sample_method=:hutch (default behavior)
        mc = MeanCompliance(problem, solver; method=:trace, nv=5, sample_method=:hutch)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
    end

    @testset "Branch 2: method=:approx creates TraceEstimationMean" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test with default parameters
        mc = MeanCompliance(problem, solver; method=:approx)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean

        # Test with nv specified
        mc = MeanCompliance(problem, solver; method=:approx, nv=7)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 7
    end

    @testset "Branch 3: method=:svd_trace creates TraceEstimationSVDMean" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test with default parameters
        mc = MeanCompliance(problem, solver; method=:svd_trace)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean

        # Test with nv specified
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=4)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test size(mc.method.V, 2) == 4

        # Test with sample_method=:hadamard
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=4, sample_method=:hadamard)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test mc.method.sample_once == true  # hadamard forces sample_once=true

        # Test with sample_method=:hutch
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=4, sample_method=:hutch)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
    end

    @testset "V matrix parameter handling - TraceEstimationMean" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test V === nothing with nv === nothing (default nv=1)
        mc = MeanCompliance(problem, solver; method=:trace)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 1

        # Test V === nothing with nv specified
        mc = MeanCompliance(problem, solver; method=:trace, nv=10)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 10

        # Test V provided with nv === nothing (use size(V, 2))
        V_provided = zeros(Float64, size(F, 2), 8)
        mc = MeanCompliance(problem, solver; method=:trace, V=V_provided)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 8

        # Test V provided with nv specified (take first nv columns)
        V_provided = zeros(Float64, size(F, 2), 15)
        mc = MeanCompliance(problem, solver; method=:trace, V=V_provided, nv=5)
        @test mc.method isa TopOpt.Functions.TraceEstimationMean
        @test size(mc.method.V, 2) == 5
    end

    @testset "V matrix parameter handling - TraceEstimationSVDMean" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test V === nothing with nv === nothing (default nv=1)
        mc = MeanCompliance(problem, solver; method=:svd_trace)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test size(mc.method.V, 2) == 1

        # Test V === nothing with nv specified
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=10)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test size(mc.method.V, 2) == 10

        # Test V provided with nv === nothing (use size(V, 2))
        # Note: V must match US dimensions, not F dimensions
        # Get US from ExactSVDMean to determine correct size
        exact_svd = TopOpt.Functions.ExactSVDMean(F)
        nv_us = size(exact_svd.US, 2)
        V_provided = zeros(Float64, nv_us, 8)
        mc = MeanCompliance(problem, solver; method=:svd_trace, V=V_provided)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test size(mc.method.V, 2) == 8

        # Test V provided with nv specified (take first nv columns)
        V_provided = zeros(Float64, nv_us, 12)
        mc = MeanCompliance(problem, solver; method=:svd_trace, V=V_provided, nv=6)
        @test mc.method isa TopOpt.Functions.TraceEstimationSVDMean
        @test size(mc.method.V, 2) == 6
    end

    @testset "sample_method symbol conversion" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test hadamard symbol conversion for TraceEstimationMean
        mc = MeanCompliance(problem, solver; method=:trace, nv=3, sample_method=:hadamard)
        @test mc.method.sample_method === TopOpt.Functions.hadamard!

        # Test hutch symbol conversion for TraceEstimationMean
        mc = MeanCompliance(problem, solver; method=:trace, nv=3, sample_method=:hutch)
        @test mc.method.sample_method === TopOpt.Functions.hutch_rand!

        # Test hadamard symbol conversion for TraceEstimationSVDMean
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=3, sample_method=:hadamard)
        @test mc.method.sample_method === TopOpt.Functions.hadamard!

        # Test hutch symbol conversion for TraceEstimationSVDMean
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=3, sample_method=:hutch)
        @test mc.method.sample_method === TopOpt.Functions.hutch_rand!
    end

    @testset "sample_once parameter handling" begin
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))

        # Test sample_once=true for TraceEstimationMean
        mc = MeanCompliance(problem, solver; method=:trace, nv=3, sample_once=true)
        @test mc.method.sample_once == true

        # Test sample_once=false for TraceEstimationMean
        mc = MeanCompliance(problem, solver; method=:trace, nv=3, sample_once=false)
        @test mc.method.sample_once == false

        # Test sample_once=true for TraceEstimationSVDMean
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=3, sample_once=true)
        @test mc.method.sample_once == true

        # Test sample_once=false for TraceEstimationSVDMean
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=3, sample_once=false)
        @test mc.method.sample_once == false

        # Test that hadamard forces sample_once=true for TraceEstimationSVDMean
        mc = MeanCompliance(problem, solver; method=:svd_trace, nv=3, sample_method=:hadamard, sample_once=false)
        @test mc.method.sample_once == true
    end

    @testset "Both branches produce valid function values" begin
        # Verify that both branches produce valid, positive compliance values
        x = fill(0.5, nels[1] * nels[2])

        # Test TraceEstimationMean
        solver_trace = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace = MeanCompliance(problem, solver_trace; method=:trace, nv=10)
        val_trace = mc_trace(PseudoDensities(x))
        @test val_trace > 0
        @test isfinite(val_trace)

        # Test TraceEstimationSVDMean
        solver_svd = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_svd = MeanCompliance(problem, solver_svd; method=:svd_trace, nv=10)
        val_svd = mc_svd(PseudoDensities(x))
        @test val_svd > 0
        @test isfinite(val_svd)
    end

    @testset "Both branches produce consistent gradients" begin
        x = fill(0.5, nels[1] * nels[2])

        # Test TraceEstimationMean gradient
        solver_trace = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace = MeanCompliance(problem, solver_trace; method=:trace, nv=10)
        grad_trace = Zygote.gradient(x -> mc_trace(PseudoDensities(x)), x)[1]
        @test length(grad_trace) == length(x)
        @test all(isfinite.(grad_trace))
        @test mean(grad_trace) < 0  # Gradient should be negative (more material = lower compliance)

        # Test TraceEstimationSVDMean gradient
        solver_svd = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_svd = MeanCompliance(problem, solver_svd; method=:svd_trace, nv=10)
        grad_svd = Zygote.gradient(x -> mc_svd(PseudoDensities(x)), x)[1]
        @test length(grad_svd) == length(x)
        @test all(isfinite.(grad_svd))
        @test mean(grad_svd) < 0
    end

    @testset "Code duplication verification - both branches have identical V handling" begin
        # This test verifies that the duplicated code in both branches
        # produces identical results for equivalent inputs

        # Test case 1: V === nothing, nv === nothing
        solver1 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace1 = MeanCompliance(problem, solver1; method=:trace)
        mc_svd1 = MeanCompliance(problem, solver1; method=:svd_trace)
        @test size(mc_trace1.method.V, 2) == size(mc_svd1.method.V, 2) == 1

        # Test case 2: V === nothing, nv=5
        solver2 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace2 = MeanCompliance(problem, solver2; method=:trace, nv=5)
        mc_svd2 = MeanCompliance(problem, solver2; method=:svd_trace, nv=5)
        @test size(mc_trace2.method.V, 2) == size(mc_svd2.method.V, 2) == 5

        # Test case 3: V provided with nv === nothing
        exact_svd = TopOpt.Functions.ExactSVDMean(F)
        nv_us = size(exact_svd.US, 2)
        V_trace = zeros(Float64, size(F, 2), 6)
        V_svd = zeros(Float64, nv_us, 6)

        solver3 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace3 = MeanCompliance(problem, solver3; method=:trace, V=V_trace)
        mc_svd3 = MeanCompliance(problem, solver3; method=:svd_trace, V=V_svd)
        @test size(mc_trace3.method.V, 2) == size(mc_svd3.method.V, 2) == 6

        # Test case 4: V provided with nv specified
        V_trace = zeros(Float64, size(F, 2), 10)
        V_svd = zeros(Float64, nv_us, 10)

        solver4 = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(2.0))
        mc_trace4 = MeanCompliance(problem, solver4; method=:trace, V=V_trace, nv=4)
        mc_svd4 = MeanCompliance(problem, solver4; method=:svd_trace, V=V_svd, nv=4)
        @test size(mc_trace4.method.V, 2) == size(mc_svd4.method.V, 2) == 4
    end
end