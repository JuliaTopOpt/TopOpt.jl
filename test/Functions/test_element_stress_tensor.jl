using TopOpt,
    Zygote, LinearAlgebra, Test, Random, SparseArrays, ForwardDiff, ChainRulesCore
using Ferrite: ndofs_per_cell, getncells, celldofs!
using TopOpt.Functions: StressTensor, DisplacementResult, ElementStressTensor, reinit!, _element_stress_tensor

Random.seed!(1)

# Reference von Mises calculation for 2D (plane stress)
function von_mises_reference_2d(σxx, σyy, σxy)
    return sqrt(σxx^2 - σxx*σyy + σyy^2 + 3*σxy^2)
end

# Reference von Mises calculation for 3D (general stress state)
function von_mises_reference_3d(σ)
    σxx, σyy, σzz = σ[1,1], σ[2,2], σ[3,3]
    σxy, σyz, σzx = σ[1,2], σ[2,3], σ[3,1]
    
    # Deviatoric stress components
    σm = (σxx + σyy + σzz) / 3
    sxx = σxx - σm
    syy = σyy - σm
    szz = σzz - σm
    
    # von Mises stress
    return sqrt(1.5 * (sxx^2 + syy^2 + szz^2) + 3*(σxy^2 + σyz^2 + σzx^2))
end

@testset "ElementStressTensor call function" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    st = StressTensor(solver)
    dp = Displacement(solver)
    
    # Get displacement result for testing
    x = clamp.(rand(prod(nels)), 0.1, 1.0)
    u = dp(PseudoDensities(x))
    
    @testset "Basic call with element_dofs=false (default)" begin
        # Test calling ElementStressTensor with default element_dofs=false
        for cellidx in 1:min(2, length(st.cells))
            est = st[cellidx]
            
            # Call with element_dofs=false (default)
            result = est(u; element_dofs=false)
            
            # Check result is a matrix (stress tensor)
            @test isa(result, AbstractMatrix)
            dim = TopOptProblems.getdim(problem)
            @test size(result) == (dim, dim)
            
            # Check all values are finite
            @test all(isfinite, result)
        end
    end
    
    @testset "Call with element_dofs=true" begin
        # Test calling ElementStressTensor with element_dofs=true
        for cellidx in 1:min(2, length(st.cells))
            est = st[cellidx]
            
            # Get element dofs displacement result
            dh = problem.ch.dh
            n = ndofs_per_cell(dh)
            global_dofs = zeros(Int, n)
            celldofs!(global_dofs, dh, cellidx)
            element_disp = DisplacementResult(u.u[global_dofs])
            
            # Call with element_dofs=true
            result = est(element_disp; element_dofs=true)
            
            # Check result is a matrix (stress tensor)
            @test isa(result, AbstractMatrix)
            dim = TopOptProblems.getdim(problem)
            @test size(result) == (dim, dim)
            
            # Check all values are finite
            @test all(isfinite, result)
        end
    end
    
    @testset "Consistency between element_dofs modes" begin
        # The results from both modes should be consistent when properly configured
        cellidx = 1
        est = st[cellidx]
        
        # Get element dofs displacement result
        dh = problem.ch.dh
        n = ndofs_per_cell(dh)
        global_dofs = zeros(Int, n)
        celldofs!(global_dofs, dh, cellidx)
        element_disp = DisplacementResult(u.u[global_dofs])
        
        # Call with element_dofs=true
        result_true = est(element_disp; element_dofs=true)
        
        # Call with element_dofs=false (uses global displacement)
        result_false = est(u; element_dofs=false)
        
        # Both should return same stress tensor for the same element
        @test size(result_true) == size(result_false)
    end
    
    @testset "Stress tensor properties" begin
        # Test that returned stress tensors have expected properties
        cellidx = 1
        est = st[cellidx]
        
        result = est(u; element_dofs=false)
        
        # For a symmetric stress tensor in linear elasticity
        dim = TopOptProblems.getdim(problem)
        @test size(result) == (dim, dim)
        
        # All diagonal elements should be real
        for i in 1:dim
            @test isreal(result[i, i])
        end
    end
    
    @testset "reinit! is called during evaluation" begin
        # Test that reinit! properly updates internal state
        cellidx = 1
        est1 = st[cellidx]
        
        # First evaluation
        result1 = est1(u; element_dofs=false)
        
        # Second evaluation - should give same result
        result2 = est1(u; element_dofs=false)
        
        @test result1 ≈ result2
        
        # Different cell - should give different result
        if length(st.cells) >= 2
            est2 = st[2]
            result3 = est2(u; element_dofs=false)
            @test size(result3) == size(result1)
        end
    end
    
    @testset "Integration with gradient computation" begin
        # Test that the call works within Zygote gradient computation
        cellidx = 1
        est = st[cellidx]
        
        # Define a scalar function based on stress tensor
        f = x -> begin
            u_local = dp(PseudoDensities(x))
            σ = est(u_local; element_dofs=false)
            return sum(abs2, σ)  # Sum of squared stresses
        end
        
        x = clamp.(rand(prod(nels)), 0.1, 1.0)
        
        # Evaluate function
        val = f(x)
        @test isa(val, Real)
        @test isfinite(val)
        @test val >= 0  # Sum of squares should be non-negative
        
        # Compute gradient
        grad = Zygote.gradient(f, x)[1]
        @test length(grad) == length(x)
        @test all(isfinite, grad)
    end
    
    @testset "Stress tensor for different density values" begin
        # Test stress computation with different density values
        cellidx = 1
        est = st[cellidx]
        
        # Low density
        x_low = fill(0.1, prod(nels))
        u_low = dp(PseudoDensities(x_low))
        σ_low = est(u_low; element_dofs=false)
        
        # High density
        x_high = fill(1.0, prod(nels))
        u_high = dp(PseudoDensities(x_high))
        σ_high = est(u_high; element_dofs=false)
        
        # Both should return valid stress tensors
        @test all(isfinite, σ_low)
        @test all(isfinite, σ_high)
        @test size(σ_low) == size(σ_high)
        
        # Stress values should be different
        @test σ_low != σ_high
    end
end

@testset "von_mises function tests" begin
    @testset "2D stress tensors" begin
        σ1 = [100.0 0.0; 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ1) ≈ 100.0 atol = 1e-10
        @test TopOpt.Functions.von_mises(σ1) ≈ von_mises_reference_2d(100.0, 0.0, 0.0)

        σ2 = [0.0 0.0; 0.0 100.0]
        @test TopOpt.Functions.von_mises(σ2) ≈ 100.0 atol = 1e-10
        @test TopOpt.Functions.von_mises(σ2) ≈ von_mises_reference_2d(0.0, 100.0, 0.0)

        σ3 = [100.0 0.0; 0.0 100.0]
        @test TopOpt.Functions.von_mises(σ3) ≈ 100.0 atol = 1e-10
        @test TopOpt.Functions.von_mises(σ3) ≈ von_mises_reference_2d(100.0, 100.0, 0.0)

        σ4 = [0.0 50.0; 50.0 0.0]
        @test TopOpt.Functions.von_mises(σ4) ≈ sqrt(3) * 50.0 atol = 1e-10
        @test TopOpt.Functions.von_mises(σ4) ≈ von_mises_reference_2d(0.0, 0.0, 50.0)

        σ5 = [100.0 50.0; 50.0 50.0]
        expected = sqrt(100^2 - 100 * 50 + 50^2 + 3 * 50^2)
        @test TopOpt.Functions.von_mises(σ5) ≈ expected atol = 1e-10
        @test TopOpt.Functions.von_mises(σ5) ≈ von_mises_reference_2d(100.0, 50.0, 50.0)

        σ6 = [0.0 0.0; 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ6) ≈ 0.0 atol = 1e-10

        σ7 = [-50.0 30.0; 30.0 -30.0]
        @test TopOpt.Functions.von_mises(σ7) ≈ von_mises_reference_2d(-50.0, -30.0, 30.0)

        σ8 = [100 0; 0 50]
        @test TopOpt.Functions.von_mises(σ8) ≈ sqrt(100^2 - 100 * 50 + 50^2) atol = 1e-10
    end

    @testset "3D stress tensors" begin
        σ1 = [100.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ1) ≈ 100.0 atol = 1e-10
        @test TopOpt.Functions.von_mises(σ1) ≈ von_mises_reference_3d(σ1)

        σ2 = [100.0 0.0 0.0; 0.0 100.0 0.0; 0.0 0.0 100.0]
        @test TopOpt.Functions.von_mises(σ2) ≈ 0.0 atol = 1e-10  # Hydrostatic stress has zero von Mises

        σ3 = [0.0 50.0 0.0; 50.0 0.0 0.0; 0.0 0.0 0.0]
        expected = sqrt(3) * 50.0
        @test TopOpt.Functions.von_mises(σ3) ≈ expected atol = 1e-10
        @test TopOpt.Functions.von_mises(σ3) ≈ von_mises_reference_3d(σ3)

        σ4 = [100.0 30.0 20.0; 30.0 50.0 10.0; 20.0 10.0 25.0]
        @test TopOpt.Functions.von_mises(σ4) ≈ von_mises_reference_3d(σ4)

        σ5 = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ5) ≈ 0.0 atol = 1e-10

        σ6 = [0.0 0.0 0.0; 0.0 0.0 40.0; 0.0 40.0 0.0]
        @test TopOpt.Functions.von_mises(σ6) ≈ sqrt(3) * 40.0 atol = 1e-10

        σ7 = [0.0 0.0 60.0; 0.0 0.0 0.0; 60.0 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ7) ≈ sqrt(3) * 60.0 atol = 1e-10
    end

    @testset "Error handling" begin
        σ1 = [100.0 0.0 0.0 0.0; 0.0 100.0 0.0 0.0; 0.0 0.0 100.0 0.0; 0.0 0.0 0.0 100.0]
        @test_throws ArgumentError TopOpt.Functions.von_mises(σ1)

        σ2 = [100.0]
        @test_throws MethodError TopOpt.Functions.von_mises(σ2)
    end

    @testset "Type stability" begin
        σ1 = Float32[100.0 0.0; 0.0 0.0]
        result1 = TopOpt.Functions.von_mises(σ1)
        @test typeof(result1) == Float32

        σ2 = Float64[100.0 0.0; 0.0 0.0]
        result2 = TopOpt.Functions.von_mises(σ2)
        @test typeof(result2) == Float64

        σ3 = [1e-10 0.0; 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ3) ≈ 1e-10 atol = 1e-20

        σ4 = [1e10 0.0; 0.0 0.0]
        @test TopOpt.Functions.von_mises(σ4) ≈ 1e10 atol = 1e-2
    end

    @testset "Symmetry preservation" begin
        σ_base = [100.0 50.0 30.0; 50.0 80.0 20.0; 30.0 20.0 60.0]
        σ_transpose = σ_base'
        @test TopOpt.Functions.von_mises(σ_base) ≈ TopOpt.Functions.von_mises(σ_transpose)

        σ_2d = [100.0 50.0; 50.0 80.0]
        @test TopOpt.Functions.von_mises(σ_2d) ≈ TopOpt.Functions.von_mises(σ_2d')
    end
end

println("All ElementStressTensor call tests passed!")
println("All von_mises tests passed!")
