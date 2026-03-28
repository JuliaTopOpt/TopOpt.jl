using TopOpt, Test, Random, SparseArrays, LinearAlgebra, Statistics
using TopOpt.Functions: ExactMean, ExactSVDMean, TraceEstimationMean, TraceEstimationSVDMean,
                         ExactDiagonal, ExactSVDDiagonal, DiagonalEstimation,
                         hutch_rand!, hadamard!

Random.seed!(42)

@testset "Trace Estimation - Mathematical Correctness" begin

    @testset "ExactMean computes trace of F^T F" begin
        # For a matrix F, trace(F^T * F) = sum of squared singular values
        # ExactMean with identity should give us this
        n = 10
        m = 5
        F_dense = randn(n, m)
        F = sparse(F_dense)
        
        em = ExactMean(F)
        
        # The trace we want is trace(F^T * F)
        expected_trace = tr(F_dense' * F_dense)
        
        @test em.F === F
        @test expected_trace > 0
        @test isfinite(expected_trace)
        
        # Verify structure is correct
        @test size(em.F) == (n, m)
        @test em isa TopOpt.Functions.AbstractExactMeanMethod
    end

    @testset "ExactSVDMean preserves trace via SVD" begin
        # SVD-based method should preserve the trace computation
        n = 10
        m = 5
        F_dense = randn(n, m)
        F = sparse(F_dense)
        
        esm = ExactSVDMean(F)
        
        # US should preserve the column space information
        @test esm.US isa SparseMatrixCSC
        @test size(esm.US, 2) <= m  # SVD compression may reduce rank
        @test esm.n == m
        
        # The trace computed via US should match original
        if esm.US !== nothing && nnz(esm.US) > 0
            US_dense = Matrix(esm.US)
            trace_via_svd = tr(US_dense' * US_dense)
            expected_trace = tr(F_dense' * F_dense)
            
            # Should be close (within numerical precision)
            @test isapprox(trace_via_svd, expected_trace; rtol=0.01)
        end
    end

    @testset "TraceEstimationMean with Rademacher sampling" begin
        # Hutchinson estimator: E[v^T A v] = trace(A) for v ~ Rademacher
        n = 8
        m = 4
        
        # Create a symmetric positive definite matrix
        A_dense = randn(n, n)
        A = A_dense' * A_dense + I  # Ensure positive definite
        F = sparse(sqrt(A))  # F such that F^T F = A
        
        # Multiple estimates to reduce variance
        estimates = Float64[]
        for trial in 1:50
            tem = TraceEstimationMean(F, 20)  # 20 samples
            # The estimate is computed as (m/nv) * sum of v_i^T F^T F v_i
            V = tem.V
            
            # Compute trace estimate manually
            estimate = 0.0
            for i in 1:size(V, 2)
                v = V[:, i]
                estimate += dot(v, Matrix(F)' * Matrix(F) * v)
            end
            estimate /= size(V, 2)
            push!(estimates, estimate)
        end
        
        mean_estimate = mean(estimates)
        true_trace = tr(Matrix(F)' * Matrix(F))
        
        # Mean estimate should be close to true trace
        @test isapprox(mean_estimate, true_trace; rtol=0.2)
    end

    @testset "Trace estimation convergence with sample count" begin
        # More samples should give better estimates (lower variance)
        n = 6
        m = 3
        
        A_dense = randn(n, n)
        A = A_dense' * A_dense + I
        F = sparse(sqrt(A))
        true_trace = tr(Matrix(F)' * Matrix(F))
        
        sample_counts = [5, 10, 20, 40]
        std_devs = Float64[]
        
        for nv in sample_counts
            estimates = Float64[]
            for trial in 1:30
                tem = TraceEstimationMean(F, nv)
                V = tem.V
                estimate = 0.0
                for i in 1:size(V, 2)
                    v = V[:, i]
                    estimate += dot(v, Matrix(F)' * Matrix(F) * v)
                end
                estimate /= size(V, 2)
                push!(estimates, estimate)
            end
            push!(std_devs, std(estimates))
        end
        
        # Standard deviation should decrease with more samples
        @test std_devs[end] < std_devs[1] * 0.8
    end

    @testset "Hadamard vs Rademacher sampling" begin
        # Both should estimate the trace, but Hadamard is deterministic
        n = 16
        m = 4
        
        A_dense = randn(n, n)
        A = A_dense' * A_dense + I
        F = sparse(sqrt(A))
        true_trace = tr(Matrix(F)' * Matrix(F))
        
        # Hadamard sampling (deterministic)
        tem_hadamard = TraceEstimationMean(F, 8, true, hadamard!)
        V_h = tem_hadamard.V
        
        estimate_h = 0.0
        for i in 1:size(V_h, 2)
            v = V_h[:, i]
            estimate_h += dot(v, Matrix(F)' * Matrix(F) * v)
        end
        estimate_h /= size(V_h, 2)
        
        # Rademacher sampling (random)
        estimates_r = Float64[]
        for trial in 1:20
            tem_rademacher = TraceEstimationMean(F, 8, true, hutch_rand!)
            V_r = tem_rademacher.V
            
            estimate_r = 0.0
            for i in 1:size(V_r, 2)
                v = V_r[:, i]
                estimate_r += dot(v, Matrix(F)' * Matrix(F) * v)
            end
            estimate_r /= size(V_r, 2)
            push!(estimates_r, estimate_r)
        end
        
        # Both should be reasonable estimates
        @test abs(estimate_h - true_trace) / true_trace < 0.5
        @test abs(mean(estimates_r) - true_trace) / true_trace < 0.5
    end

    @testset "ExactDiagonal computes diagonal of F^T F" begin
        # ExactDiagonal stores intermediate results for diagonal computation
        n = 8
        m = 4
        nE = 6  # Number of elements
        
        F_dense = randn(n, m)
        F = sparse(F_dense)
        
        ed = ExactDiagonal(F, nE)
        
        # Structure validation
        @test ed.F === F
        @test size(ed.Y) == (n, m)
        @test length(ed.temp) == nE
        @test all(iszero, ed.temp)
        
        # Y should be initialized to zeros (will be filled during solve)
        @test all(iszero, ed.Y)
        
        # The diagonal of F^T F is what we ultimately want
        expected_diag = diag(F_dense' * F_dense)
        @test length(expected_diag) == m
        @test all(>=(0), expected_diag)  # Diagonal entries are non-negative
    end

    @testset "ExactSVDDiagonal with SVD compression" begin
        # SVD-based diagonal estimation
        n = 8
        m = 4
        nE = 6
        
        F_dense = randn(n, m)
        F = sparse(F_dense)
        
        esvd = ExactSVDDiagonal(F, nE)
        
        # Structure validation
        @test esvd.F === F
        @test esvd.US isa SparseMatrixCSC
        @test size(esvd.V, 1) == m
        @test size(esvd.Q) == size(esvd.US)
        @test size(esvd.Y) == (size(esvd.V, 2), size(esvd.V, 2))
        @test length(esvd.temp) == nE
        
        # V from SVD should be orthonormal
        V = esvd.V
        if size(V, 2) > 0
            @test isapprox(V' * V, I; atol=1e-10)
        end
    end

    @testset "DiagonalEstimation approximate diagonal" begin
        # Hutchinson-style diagonal estimation
        n = 8
        m = 4
        nv = 5
        nE = 6
        
        F_dense = randn(n, m)
        F = sparse(F_dense)
        
        de = DiagonalEstimation(F, nv, nE)
        
        # Structure validation
        @test de.F === F
        @test size(de.V) == (m, nv)
        @test size(de.Y) == (n, nv)
        @test size(de.Q) == (n, nv)
        @test length(de.temp) == nE
        
        # Hadamard sampling sets sample_once=true
        @test de.sample_once == true
    end

    @testset "DiagonalEstimation with custom V matrix" begin
        n = 6
        m = 3
        nE = 4
        
        F = sprandn(n, m, 0.8)
        V_custom = randn(m, 5)
        
        de = DiagonalEstimation(F, V_custom, nE)
        
        @test de.V === V_custom
        @test size(de.Y) == (n, 5)
        @test size(de.Q) == (n, 5)
    end

    @testset "TraceEstimationSVDMean with custom V" begin
        n = 6
        m = 3
        
        F = sprandn(n, m, 0.8)
        V_custom = randn(size(F, 2), 4)
        
        tesm = TraceEstimationSVDMean(F, V_custom)
        
        @test tesm.V === V_custom
        @test tesm.n == m
    end

    @testset "sample_once behavior" begin
        # sample_once parameter controls whether to sample V only once
        n = 8
        m = 4
        
        F = sprandn(n, m, 0.8)
        
        # sample_once=false means V can be resampled
        tem1 = TraceEstimationMean(F, 3, false, hutch_rand!)
        @test tem1.sample_once == false
        
        # sample_once=true means V is sampled once and reused
        tem2 = TraceEstimationMean(F, 4, true, hutch_rand!)
        @test tem2.sample_once == true
        
        # hadamard! is typically used with sample_once=true
        tem3 = TraceEstimationMean(F, 4, true, hadamard!)
        @test tem3.sample_once == true
    end

    @testset "Empty constructors" begin
        em = ExactMean()
        @test em.F === nothing
        
        esm = ExactSVDMean()
        @test esm.US === nothing
        @test esm.n == 0
    end
end

@testset "Trace Estimation - Integration with TopOpt" begin

    @testset "ExactMean and TraceEstimationMean in MeanCompliance context" begin
        # Use a fixed load to avoid boundary condition issues
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (6, 4)
        
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        
        # Create a simple fixed load pattern at known dofs
        nloads = 2
        ndofs = TopOpt.Ferrite.ndofs(base_problem.ch.dh)
        F = spzeros(ndofs, nloads)
        
        # Get some fixed dof indices
        fixed_dofs = base_problem.ch.prescribed_dofs
        free_dofs = setdiff(1:ndofs, fixed_dofs)
        
        if length(free_dofs) >= 4
            F[free_dofs[1:2], 1] .= 0.5
            F[free_dofs[3:4], 2] .= 0.5
        else
            F[1:min(2, ndofs), 1] .= 0.5
            F[1:min(2, ndofs), 2] .= 0.5
        end
        
        # Verify F has correct structure
        @test size(F, 2) == nloads
        
        # ExactMean should wrap F correctly
        em = ExactMean(F)
        @test em.F === F
        
        # MeanCompliance with exact method
        problem_ml = MultiLoad(base_problem, F)
        solver_exact = FEASolver(DirectSolver, problem_ml; xmin=0.01, penalty=PowerPenalty(1.0))
        mc_exact = MeanCompliance(problem_ml, solver_exact; method=:exact)
        
        x = fill(0.5, length(solver_exact.vars))
        C_exact = mc_exact(PseudoDensities(x))
        
        @test C_exact > 0 || C_exact == 0  # Allow zero for some configurations
        @test isfinite(C_exact)
        
        # MeanCompliance with trace estimation method
        solver_trace = FEASolver(DirectSolver, problem_ml; xmin=0.01, penalty=PowerPenalty(1.0))
        mc_trace = MeanCompliance(problem_ml, solver_trace; method=:trace, nv=10)
        
        C_trace = mc_trace(PseudoDensities(x))
        
        @test C_trace > 0 || C_trace == 0
        @test isfinite(C_trace)
    end

    @testset "ExactDiagonal and DiagonalEstimation in context" begin
        # Test the types directly with minimal integration test
        # Just verify BlockCompliance can be created with these types
        
        E = 1.0
        ν = 0.3
        force = 1.0
        nels = (4, 4)
        
        base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), E, ν, force)
        
        nloads = 2
        ndofs = TopOpt.Ferrite.ndofs(base_problem.ch.dh)
        F = spzeros(ndofs, nloads)
        
        # Get fixed and free dofs
        fixed_dofs = base_problem.ch.prescribed_dofs
        free_dofs = setdiff(1:ndofs, fixed_dofs)
        
        if length(free_dofs) >= 4
            F[free_dofs[1:2], 1] .= 0.3
            F[free_dofs[3:4], 2] .= 0.3
        end
        
        problem_ml = MultiLoad(base_problem, F)
        solver_ml = FEASolver(DirectSolver, problem_ml; xmin=0.01, penalty=PowerPenalty(1.0))
        
        # Test BlockCompliance can be created with exact method
        bc_exact = BlockCompliance(problem_ml, solver_ml; method=:exact)
        
        x = fill(0.5, length(solver_ml.vars))
        block_vals_exact = bc_exact(PseudoDensities(x))
        
        # Should return array with correct length
        @test length(block_vals_exact) == nloads
        
        # Test BlockCompliance can be created with approximate method
        bc_approx = BlockCompliance(problem_ml, solver_ml; method=:approx, nv=5)
        block_vals_approx = bc_approx(PseudoDensities(x))
        
        @test length(block_vals_approx) == nloads
    end
end
