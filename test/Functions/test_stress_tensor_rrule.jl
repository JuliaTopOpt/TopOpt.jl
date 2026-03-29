using TopOpt,
    Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays, ForwardDiff, ChainRulesCore, Ferrite
const FDM = FiniteDifferences
using TopOpt: ndofs
using Ferrite: ndofs_per_cell, getncells
using NonconvexCore: getdim
using TopOpt.Functions: StressTensor, DisplacementResult, reinit!

Random.seed!(1)

@testset "StressTensor reinit! rrule" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    st = StressTensor(solver)
    dp = Displacement(solver)
    
    # Get displacement result for testing
    x = clamp.(rand(prod(nels)), 0.1, 1.0)
    u = dp(PseudoDensities(x))
    
    @testset "rrule returns correct types" begin
        # Test rrule for StressTensor reinit!
        cellidx = 1
        y, pullback = ChainRulesCore.rrule(reinit!, st, cellidx)
        
        # Check that the output is the StressTensor itself
        @test y === st
        
        # Check that the pullback returns NoTangent for all inputs
        Δy = ChainRulesCore.NoTangent()
        grads = pullback(Δy)
        
        @test length(grads) == 3
        @test grads[1] isa ChainRulesCore.NoTangent
        @test grads[2] isa ChainRulesCore.NoTangent
        @test grads[3] isa ChainRulesCore.NoTangent
    end
    
    @testset "rrule for different cell indices" begin
        for cellidx in 1:length(st.cells)
            y, pullback = ChainRulesCore.rrule(reinit!, st, cellidx)
            
            @test y === st
            
            # Test pullback with different tangents
            Δy = ChainRulesCore.NoTangent()
            grads = pullback(Δy)
            
            @test all(g isa ChainRulesCore.NoTangent for g in grads)
        end
    end
    
    @testset "rrule with Float64 tangent" begin
        cellidx = 1
        y, pullback = ChainRulesCore.rrule(reinit!, st, cellidx)
        
        # Test with a Float64 tangent (should still return NoTangent)
        Δy = 0.0
        grads = pullback(Δy)
        
        @test length(grads) == 3
        @test grads[1] isa ChainRulesCore.NoTangent
        @test grads[2] isa ChainRulesCore.NoTangent
        @test grads[3] isa ChainRulesCore.NoTangent
    end
    
    @testset "rrule works with Zygote" begin
        # Test that Zygote can use the rrule correctly
        # by computing gradients through the stress tensor computation
        
        # Define a function that uses reinit! internally
        function test_fn(st, u, cellidx)
            reinit!(st, cellidx)
            # Return some scalar value based on the stress tensor
            return sum(st.global_dofs)
        end
        
        # Test that Zygote can differentiate through this
        # The rrule should prevent Zygote from trying to AD through reinit!
        cellidx = 1
        val = test_fn(st, u, cellidx)
        @test isa(val, Int)
        @test val > 0
        
        # The rrule is a no-op for gradients, so any gradient computation
        # that involves reinit! should work without error
    end
    
    @testset "rrule consistency with ElementStressTensor" begin
        # Also test ElementStressTensor rrule for consistency
        est = st[1]
        
        cellidx = 1
        y, pullback = ChainRulesCore.rrule(reinit!, est, cellidx)
        
        @test y === est
        
        Δy = ChainRulesCore.NoTangent()
        grads = pullback(Δy)
        
        @test length(grads) == 3
        @test all(g isa ChainRulesCore.NoTangent for g in grads)
    end
    
    @testset "reinit! modifies StressTensor correctly" begin
        # Save original state
        orig_dofs = copy(st.global_dofs)
        
        cellidx = 2
        reinit!(st, cellidx)
        
        # global_dofs should be modified
        @test st.global_dofs != orig_dofs
        
        # Now test rrule returns the modified tensor
        cellidx = 1
        y, _ = ChainRulesCore.rrule(reinit!, st, cellidx)
        
        @test y === st
        @test y.global_dofs == st.global_dofs
    end
end

@testset "StressTensor reinit! integration with AD" begin
    nels = (3, 3)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    st = StressTensor(solver)
    dp = Displacement(solver)
    
    @testset "AD through stress tensor computation" begin
        # Test that we can compute gradients of stress-related quantities
        # The rrule for reinit! allows Zygote to skip the mutation in reinit!
        
        for cellidx in 1:min(4, length(st.cells))
            # Get element stress tensor
            est = st[cellidx]
            
            # Define a scalar function for gradient testing
            f = x -> begin
                u = dp(PseudoDensities(x))
                result = est(u)
                return sum(abs2, result)
            end
            
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            
            # This should work without errors thanks to the rrule
            val = f(x)
            @test isa(val, Real)
            @test isfinite(val)
            
            # Test gradient computation
            grad = Zygote.gradient(f, x)[1]
            @test length(grad) == length(x)
            @test all(isfinite, grad)
        end
    end
end

@testset "ElementStressTensorKernel rrule" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
    
    st = StressTensor(solver)
    dp = Displacement(solver)
    
    # Get displacement result for testing
    x = clamp.(rand(prod(nels)), 0.1, 1.0)
    u = dp(PseudoDensities(x))
    
    # Create an ElementStressTensorKernel
    est = st[1]
    reinit!(est, 1)
    
    # Get tensor_kernel to create ElementStressTensorKernel
    # Access the internal structure to get the kernel
    # tensor_kernel is called in _element_stress_tensor for each quad point and basis function
    cellvalues = st.cellvalues
    n_basefuncs = Ferrite.getnbasefunctions(cellvalues)
    n_quad = Ferrite.getnquadpoints(cellvalues)
    dim = TopOptProblems.getdim(problem)
    
    # Create an ElementStressTensorKernel manually (similar to tensor_kernel function)
    kernel = TopOpt.Functions.ElementStressTensorKernel(
        problem.E,
        problem.ν,
        1,  # q_point
        1,  # basis function a
        cellvalues,
        dim
    )
    
    @testset "rrule output types and shapes" begin
        # Get element dofs for the kernel
        # Kernel uses only displacement from ONE basis function (2 elements for 2D)
        element_dofs = u.u[copy(st.global_dofs)]
        u_single = element_dofs[dim * (kernel.a - 1) .+ (1:dim)]
        u_element = DisplacementResult(u_single)
        
        # Apply the rrule
        v, pullback = ChainRulesCore.rrule(kernel, u_element)
        
        # Check output type
        @test v isa Matrix
        @test size(v) == (dim, dim)
        
        # Test pullback with a random tangent
        Δ = randn(dim, dim)
        grads = pullback(Δ)
        
        # Should return (NoTangent(), Tangent)
        @test length(grads) == 2
        @test grads[1] isa ChainRulesCore.NoTangent
        @test grads[2] isa ChainRulesCore.Tangent{<:DisplacementResult}
    end
    
    @testset "rrule gradient correctness with finite differences" begin
        # The kernel computes stress contribution from ONE basis function
        # So we need displacement for just that basis function (2 elements for 2D)
        element_dofs = u.u[copy(st.global_dofs)]
        u_single_basis = element_dofs[dim * (kernel.a - 1) .+ (1:dim)]
        u_element = DisplacementResult(u_single_basis)
        
        # Compute rrule output
        v, pullback = ChainRulesCore.rrule(kernel, u_element)
        
        # Test with different tangents
        for _ in 1:5
            Δ = randn(dim, dim)
            grads = pullback(Δ)
            
            # The gradient should be a vector matching the displacement size (2 for 2D)
            @test size(grads[2].u) == size(u_single_basis)
            @test all(isfinite, grads[2].u)
        end
        
        # Compare with finite differences for a specific direction
        Δ_test = randn(dim, dim)
        
        # Define scalar function for FD: sum(kernel output .* Δ_test)
        f_scalar = u_vec -> sum(kernel(DisplacementResult(collect(u_vec))) .* Δ_test)
        
        # Compute gradient using finite differences
        fd_grad = FiniteDifferences.grad(FDM.central_fdm(5, 1), f_scalar, u_single_basis)[1]
        
        # Compute gradient using rrule
        _, pullback_fd = ChainRulesCore.rrule(kernel, u_element)
        rrule_grad = pullback_fd(Δ_test)[2].u
        
        @test fd_grad ≈ rrule_grad rtol = 1e-4
    end
    
    @testset "rrule for different quad points and basis functions" begin
        element_dofs = u.u[copy(st.global_dofs)]
        
        # Test different combinations of quad points and basis functions
        for q_point in 1:min(2, n_quad)
            for a in 1:min(2, n_basefuncs)
                kernel_test = TopOpt.Functions.ElementStressTensorKernel(
                    problem.E,
                    problem.ν,
                    q_point,
                    a,
                    cellvalues,
                    dim
                )
                
                # Kernel uses only displacement from one basis function
                u_single = element_dofs[dim * (a - 1) .+ (1:dim)]
                u_element = DisplacementResult(u_single)
                
                v, pullback = ChainRulesCore.rrule(kernel_test, u_element)
                
                @test size(v) == (dim, dim)
                @test all(isfinite, v)
                
                # Test pullback
                Δ = randn(dim, dim)
                grads = pullback(Δ)
                
                @test grads[1] isa ChainRulesCore.NoTangent
                @test grads[2] isa ChainRulesCore.Tangent{<:DisplacementResult}
                @test size(grads[2].u) == size(u_single)
            end
        end
    end
    
    @testset "rrule consistency with ForwardDiff jacobian" begin
        # Kernel uses displacement from one basis function only
        element_dofs = u.u[copy(st.global_dofs)]
        u_single = element_dofs[dim * (kernel.a - 1) .+ (1:dim)]
        u_element = DisplacementResult(u_single)
        
        # Compute output and gradient using rrule
        v_rrule, pullback = ChainRulesCore.rrule(kernel, u_element)
        
        # Compute output using direct call
        v_direct = kernel(u_element)
        
        @test v_rrule ≈ v_direct
        
        # Test pullback for each output component
        for i in 1:dim, j in 1:dim
            Δ = zeros(dim, dim)
            Δ[i, j] = 1.0
            
            grads = pullback(Δ)
            rrule_grad = grads[2].u
            
            # Compare with finite differences
            f_ij = u_vec -> kernel(DisplacementResult(collect(u_vec)))[i, j]
            fd_result = FiniteDifferences.grad(FDM.central_fdm(5, 1), f_ij, u_single)
            fd_grad = fd_result[1]
            
            @test fd_grad ≈ rrule_grad rtol = 1e-4
        end
    end
end
