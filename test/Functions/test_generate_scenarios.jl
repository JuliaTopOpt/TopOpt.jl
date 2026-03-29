using Test, SparseArrays, LinearAlgebra, Random

# Import the function - assuming it's exported from TopOpt
using TopOpt: generate_scenarios

@testset "generate_scenarios - Basic Functionality" begin

    @testset "Output structure validation" begin
        # Test basic output properties
        dof = 5
        sz = (10, 3)
        f = 1.0
        
        result = generate_scenarios(dof, sz, f)
        
        @test result isa SparseMatrixCSC{Float64, Int}
        @test size(result) == (10, 3)
        @test nnz(result) == 3  # One non-zero per column
    end

    @testset "Non-zero entries at correct positions" begin
        # Verify all non-zeros are in the specified dof row
        dof = 3
        sz = (8, 5)
        f = 2.5
        
        result = generate_scenarios(dof, sz, f)
        
        # Find non-zero positions
        rows, cols, vals = findnz(result)
        
        # All non-zeros should be in row 'dof'
        @test all(rows .== dof)
        
        # Should have one entry per column
        @test length(cols) == 5
        @test sort(cols) == [1, 2, 3, 4, 5]
    end

    @testset "Base force value without perturbation" begin
        # Test with zero perturbation to verify base values
        dof = 2
        sz = (6, 4)
        f = 10.0
        
        # Zero perturbation
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        rows, cols, vals = findnz(result)
        
        # All values should equal f exactly
        @test all(vals .== f)
        @test length(vals) == 4
    end

    @testset "Correct dimensions for various sizes" begin
        test_cases = [
            (1, (5, 1)),      # Single scenario
            (3, (10, 5)),     # Multiple scenarios
            (7, (20, 10)),    # Larger matrix
            (1, (1, 1)),      # Minimal case
        ]
        
        for (dof, sz) in test_cases
            result = generate_scenarios(dof, sz, 1.0)
            @test size(result) == sz
            @test nnz(result) == sz[2]
        end
    end
end

@testset "generate_scenarios - Perturbation Behavior" begin

    @testset "Default perturbation produces variation" begin
        # Default perturb is () -> (rand() - 0.5), giving range [-0.5, 0.5]
        Random.seed!(123)
        dof = 1
        sz = (5, 100)  # Many scenarios to test statistical properties
        f = 1.0
        
        result = generate_scenarios(dof, sz, f)
        rows, cols, vals = findnz(result)
        
        # Values should vary (not all identical)
        @test length(unique(vals)) > 1
        
        # All values should be positive since f > 0 and perturb > -0.5
        @test all(vals .> 0)
        
        # Values should be in range [0.5*f, 1.5*f]
        @test all(vals .>= 0.5 * f)
        @test all(vals .<= 1.5 * f)
    end

    @testset "Custom perturbation function" begin
        # Test with custom perturbation
        dof = 2
        sz = (4, 3)
        f = 5.0
        
        # Fixed perturbation
        fixed_perturb = () -> 0.2
        result = generate_scenarios(dof, sz, f, fixed_perturb)
        
        rows, cols, vals = findnz(result)
        
        # All values should be f * (1 + 0.2) = 6.0
        @test all(vals .== 6.0)
    end

    @testset "Negative perturbation" begin
        # Test with negative perturbation
        dof = 1
        sz = (3, 2)
        f = 10.0
        
        neg_perturb = () -> -0.3
        result = generate_scenarios(dof, sz, f, neg_perturb)
        
        rows, cols, vals = findnz(result)
        
        # Values should be f * (1 - 0.3) = 7.0
        @test all(vals .== 7.0)
    end

    @testset "Zero perturbation consistency" begin
        # With zero perturbation, all scenarios should be identical
        dof = 3
        sz = (10, 5)
        f = 2.0
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        rows, cols, vals = findnz(result)
        
        @test all(vals .== f)
        @test length(vals) == 5
    end
end

@testset "generate_scenarios - Edge Cases" begin

    @testset "Single scenario" begin
        # Single column matrix
        dof = 5
        sz = (10, 1)
        f = 1.0
        
        # Use zero perturbation for exact value check
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        @test size(result) == (10, 1)
        @test nnz(result) == 1
        @test result[dof, 1] ≈ f atol=1e-10
    end

    @testset "dof at boundaries" begin
        # Test with dof at first row
        result_first = generate_scenarios(1, (5, 3), 1.0, () -> 0.0)
        @test result_first[1, :] ≈ [1.0, 1.0, 1.0] atol=1e-10
        
        # Test with dof at last row
        result_last = generate_scenarios(5, (5, 3), 1.0, () -> 0.0)
        @test result_last[5, :] ≈ [1.0, 1.0, 1.0] atol=1e-10
    end

    @testset "Zero force" begin
        # Test with f = 0
        dof = 2
        sz = (5, 3)
        f = 0.0
        
        result = generate_scenarios(dof, sz, f)
        rows, cols, vals = findnz(result)
        
        # All values should be zero
        @test all(vals .== 0.0)
    end

    @testset "Negative force" begin
        # Test with negative f
        dof = 2
        sz = (5, 3)
        f = -5.0
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        rows, cols, vals = findnz(result)
        
        @test all(vals .== -5.0)
    end

    @testset "Large values" begin
        # Test with large force values
        dof = 1
        sz = (100, 50)
        f = 1e6
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        @test size(result) == (100, 50)
        @test nnz(result) == 50
        @test all(nonzeros(result) .== f)
    end

    @testset "Very small values" begin
        # Test with very small force values
        dof = 1
        sz = (10, 5)
        f = 1e-10
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        rows, cols, vals = findnz(result)
        @test all(vals .≈ f)
    end
end

@testset "generate_scenarios - Sparse Matrix Properties" begin

    @testset "Sparsity pattern" begin
        dof = 3
        sz = (10, 5)
        f = 1.0
        
        result = generate_scenarios(dof, sz, f)
        
        # Verify it's properly sparse
        @test issparse(result)
        
        # Check sparsity ratio
        total_elements = prod(sz)
        nonzero_elements = nnz(result)
        sparsity = 1 - nonzero_elements / total_elements
        
        @test sparsity >= 0.9  # Should be >= 90% sparse for this structure
    end

    @testset "Column-wise structure" begin
        # Each column should have exactly one non-zero
        dof = 4
        sz = (8, 6)
        f = 2.0
        
        result = generate_scenarios(dof, sz, f)
        
        for col in 1:sz[2]
            col_nnz = count(result[:, col] .!= 0)
            @test col_nnz == 1
        end
    end

    @testset "Can be converted to dense" begin
        # Should work with dense conversion for small matrices
        dof = 2
        sz = (4, 3)
        f = 1.5
        
        result = generate_scenarios(dof, sz, f)
        dense_result = Matrix(result)
        
        @test size(dense_result) == sz
        @test dense_result[dof, :] ≈ nonzeros(result) atol=1e-10
        @test sum(dense_result) ≈ sum(nonzeros(result)) atol=1e-10
    end
end

@testset "generate_scenarios - Reproducibility" begin

    @testset "Fixed random seed gives reproducible results" begin
        dof = 2
        sz = (5, 10)
        f = 1.0
        
        # First call with seed
        Random.seed!(42)
        result1 = generate_scenarios(dof, sz, f)
        
        # Second call with same seed
        Random.seed!(42)
        result2 = generate_scenarios(dof, sz, f)
        
        rows1, cols1, vals1 = findnz(result1)
        rows2, cols2, vals2 = findnz(result2)
        
        @test vals1 ≈ vals2 atol=1e-10
    end

    @testset "Different seeds give different results" begin
        dof = 2
        sz = (5, 100)
        f = 1.0
        
        # First call with seed 1
        Random.seed!(1)
        result1 = generate_scenarios(dof, sz, f)
        
        # Second call with seed 2
        Random.seed!(2)
        result2 = generate_scenarios(dof, sz, f)
        
        rows1, cols1, vals1 = findnz(result1)
        rows2, cols2, vals2 = findnz(result2)
        
        # Values should be different (statistically unlikely to be identical)
        @test vals1 != vals2
    end
end

@testset "generate_scenarios - Mathematical Properties" begin

    @testset "Sum of values" begin
        # Test that sum of values equals expected value
        dof = 1
        sz = (3, 5)
        f = 2.0
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        # Sum should be nscenarios * f
        @test sum(result) ≈ sz[2] * f atol=1e-10
    end

    @testset "Frobenius norm" begin
        # Test Frobenius norm
        dof = 2
        sz = (5, 4)
        f = 3.0
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        
        # Frobenius norm should be sqrt(nscenarios * f^2)
        expected_norm = sqrt(sz[2] * f^2)
        @test norm(result) ≈ expected_norm atol=1e-10
    end

    @testset "Transpose properties" begin
        dof = 3
        sz = (6, 4)
        f = 2.0
        
        zero_perturb = () -> 0.0
        result = generate_scenarios(dof, sz, f, zero_perturb)
        result_t = transpose(result)
        
        # Transpose should have non-zeros in column 'dof'
        @test nnz(result_t) == sz[2]
        @test all(result_t[:, dof] .== f)
    end
end

@testset "generate_scenarios - Integration Tests" begin

    @testset "Works with LinearAlgebra operations" begin
        dof = 2
        sz = (5, 3)
        f = 2.0
        
        zero_perturb = () -> 0.0
        F = generate_scenarios(dof, sz, f, zero_perturb)
        
        # Should work with matrix multiplication
        v = ones(3)
        result = F * v
        
        @test length(result) == 5
        @test result[dof] ≈ 3 * f atol=1e-10
        
        # Other entries should be zero
        for i in setdiff(1:5, dof)
            @test result[i] ≈ 0.0 atol=1e-10
        end
    end

    @testset "Can be used in sparse matrix arithmetic" begin
        dof = 1
        sz = (4, 2)
        f = 1.0
        
        zero_perturb = () -> 0.0
        F1 = generate_scenarios(dof, sz, f, zero_perturb)
        F2 = generate_scenarios(2, sz, 2.0, zero_perturb)
        
        # Addition
        F_sum = F1 + F2
        @test nnz(F_sum) == 4  # Two non-zeros per column
        
        # Scalar multiplication
        F_scaled = 3.0 * F1
        @test all(nonzeros(F_scaled) .≈ 3.0)
    end
end

@testset "generate_scenarios - Type Stability" begin

    @testset "Returns correct types" begin
        dof = 1
        sz = (3, 2)
        f = 1.0
        
        result = generate_scenarios(dof, sz, f)
        
        @test eltype(result) == Float64
        @test result isa SparseMatrixCSC{Float64, Int}
    end

    @testset "Works with different force types" begin
        dof = 1
        sz = (3, 2)
        
        # Float32 force
        f32 = Float32(1.0)
        result_f32 = generate_scenarios(dof, sz, f32)
        
        # Int force
        f_int = 5
        result_int = generate_scenarios(dof, sz, f_int)
        
        @test eltype(result_f32) == Float64  # Should convert to Float64
        @test eltype(result_int) == Float64
    end
end

println("All generate_scenarios tests completed!")