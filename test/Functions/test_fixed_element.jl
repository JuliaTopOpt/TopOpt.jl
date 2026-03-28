using Test, ChainRulesCore, Zygote, LinearAlgebra
using TopOpt.Functions: FixedElementProjector, get_fixed_element_projector, 
                        get_free_variables, get_free_variable_count
using Zygote: gradient

# Helper function for finite difference gradient check
function finite_diff_gradient(f, x::AbstractVector{T}, h=T(1e-6)) where T
    grad = similar(x)
    f_x = f(x)
    for i in eachindex(x)
        x_plus = copy(x)
        x_plus[i] += h
        grad[i] = (f(x_plus) - f_x) / h
    end
    return grad
end

@testset "FixedElementProjector" begin
    @testset "Constructor validation" begin
        # Valid construction
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        
        @test p.nel == 10
        @test get_free_variable_count(p) == 4  # elements 4-7
        @test p.free == [false, false, false, true, true, true, true, false, false, false]
        
        # Invalid: overlapping black and white
        invalid_black = falses(5); invalid_black[1:3] .= true
        invalid_white = falses(5); invalid_white[3:5] .= true  # element 3 is both
        @test_throws ArgumentError FixedElementProjector(5, invalid_black, invalid_white)
        
        # Invalid: mismatched lengths
        @test_throws ArgumentError FixedElementProjector(5, falses(5), falses(6))
    end
    
    @testset "Direct calling" begin
        # Simple case: 5 elements, no fixed
        black = falses(5)
        white = falses(5)
        p = FixedElementProjector(5, black, white)
        
        x_free = [0.0, 0.25, 0.5, 0.75, 1.0]
        ρ = p(x_free)
        
        @test length(ρ) == 5
        @test ρ == x_free  # Direct copy
    end
    
    @testset "With fixed elements" begin
        # 10 elements: 1-3 black, 8-10 white, 4-7 free
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        
        x_free = [0.0, 0.25, 0.5, 0.75]  # 4 free elements
        
        ρ = p(x_free)
        
        @test length(ρ) == 10
        @test all(ρ[1:3] .== 1.0)  # black
        @test all(ρ[8:10] .== 0.0)  # white
        @test ρ[4] == 0.0  # free element 1
        @test ρ[5] == 0.25  # free element 2
        @test ρ[6] == 0.5   # free element 3
        @test ρ[7] == 0.75  # free element 4
    end
    
    @testset "Input validation" begin
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        
        # Wrong size input
        @test_throws ArgumentError p([0.0, 1.0])  # 2 instead of 4
        @test_throws ArgumentError p(zeros(5))   # 5 instead of 4
    end
    
    @testset "Gradient preservation" begin
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        
        x_free = [0.1, 0.2, 0.3, 0.4]
        
        # Test that we can compute gradients through the projection
        f(x) = sum(p(x))
        grad = gradient(f, x_free)[1]
        
        @test length(grad) == 4
        @test all(isfinite, grad)
        
        # Finite difference check - gradient should be all ones for sum
        @test grad ≈ ones(4) rtol=1e-10
    end
end

@testset "get_fixed_element_projector (by nel)" begin
    # Empty fixed elements
    p = get_fixed_element_projector(100, Int[], Int[])
    @test get_free_variable_count(p) == 100
    @test all(p.black .== false)
    @test all(p.white .== false)
    
    # With black and white cells
    p = get_fixed_element_projector(100, 1:10, 91:100)
    @test get_free_variable_count(p) == 80
    @test count(p.black) == 10
    @test count(p.white) == 10
end

@testset "ChainRulesCore rrule" begin
    @testset "Forward pass correctness" begin
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        x_free = [0.1, 0.2, 0.3, 0.4]
        
        # Test that rrule produces same result as forward function
        y, pb = ChainRulesCore.rrule(p, x_free)
        @test y == p(x_free)
    end
    
    @testset "Pullback correctness" begin
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        x_free = [0.1, 0.2, 0.3, 0.4]
        
        y, pb = ChainRulesCore.rrule(p, x_free)
        
        # Test with random cotangent
        Δy = randn(10)  # Full size output
        Δx = pb(Δy)
        
        # Should return (NoTangent(), gradient_vector)
        @test Δx[1] isa ChainRulesCore.NoTangent
        @test length(Δx[2]) == 4
    end
    
    @testset "Gradient zero at fixed elements" begin
        # The pullback should only propagate gradients through free elements
        black = falses(5); black[1:2] .= true
        white = falses(5); white[4:5] .= true
        p = FixedElementProjector(5, black, white)  # free: only element 3
        x_free = [0.5]
        
        y, pb = ChainRulesCore.rrule(p, x_free)
        
        # Set up cotangent with non-zero values everywhere
        Δy = ones(5)
        Δx = pb(Δy)
        
        # Only the free element should contribute to gradient
        @test Δx[2][1] ≈ Δy[3] atol=1e-10
    end
    
    @testset "Integration with Zygote" begin
        black = falses(10); black[1:3] .= true
        white = falses(10); white[8:10] .= true
        p = FixedElementProjector(10, black, white)
        x_free = [0.1, 0.2, 0.3, 0.4]
        
        # Test with a composite function
        g(x) = sum(abs2, p(x))
        grad_zygote = gradient(g, x_free)[1]
        
        # Compare with finite differences
        fd_grad = finite_diff_gradient(g, x_free)
        @test grad_zygote ≈ fd_grad rtol=1e-5
    end
end

@testset "End-to-end integration" begin
    @testset "With problem" begin
        using TopOpt, TopOpt.TopOptProblems
        
        # Create a simple problem
        nels = (10, 10)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create projector with some fixed elements
        nel = prod(nels)
        black_cells = 1:10  # First row solid
        white_cells = (nel-9):nel  # Last row void
        p = get_fixed_element_projector(nel, black_cells, white_cells)
        
        # Create free design variables
        x_free = fill(0.5, get_free_variable_count(p))
        
        # Project to full density
        ρ = p(x_free)
        
        @test length(ρ) == nel
        @test all(ρ[black_cells] .== 1.0)
        @test all(ρ[white_cells] .== 0.0)
        @test all(ρ[p.free] .== 0.5)
    end
end

@testset "Edge cases" begin
    @testset "All black" begin
        black = trues(5)
        white = falses(5)
        p = FixedElementProjector(5, black, white)
        x_free = Float64[]
        ρ = p(x_free)
        @test all(ρ .== 1.0)
    end
    
    @testset "All white" begin
        black = falses(5)
        white = trues(5)
        p = FixedElementProjector(5, black, white)
        x_free = Float64[]
        ρ = p(x_free)
        @test all(ρ .== 0.0)
    end
    
    @testset "No fixed elements" begin
        black = falses(5)
        white = falses(5)
        p = FixedElementProjector(5, black, white)
        x_free = zeros(5)
        ρ = p(x_free)
        @test all(ρ .== 0.0)
    end
    
    @testset "Single element" begin
        black = falses(1)
        white = falses(1)
        p = FixedElementProjector(1, black, white)
        x_free = [0.5]
        ρ = p(x_free)
        @test ρ[1] == 0.5
    end
end

@testset "Compliance minimization with fixed elements" begin
    using TopOpt, TopOpt.TopOptProblems, Nonconvex, LinearAlgebra
    
    # Create a cantilever problem
    nels = (20, 10)
    problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    nel = prod(nels)
    
    # Define fixed elements: left side solid (black), right side void (white)
    black_cells = 1:nels[2]  # First column solid
    white_cells = (nel - nels[2] + 1):nel  # Last column void
    
    # Create projector
    free_to_full_proj = get_fixed_element_projector(nel, black_cells, white_cells)
    
    # Verify free variable count
    @test get_free_variable_count(free_to_full_proj) == nel - length(black_cells) - length(white_cells)
    
    # Create solver
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
    
    # Compliance objective
    comp = Compliance(solver)
    
    # Volume constraint
    vol = Volume(solver, fraction=true)
    
    # Initial design: all 0.5 on free variables
    x0_free = fill(0.5, get_free_variable_count(free_to_full_proj))
    
    # Objective function using projector
    function obj(x_free)
        ρ = free_to_full_proj(x_free)
        return comp(PseudoDensities(ρ))
    end
    
    # Volume constraint function
    function constr(x_free)
        ρ = free_to_full_proj(x_free)
        return vol(PseudoDensities(ρ)) - 0.5
    end
    
    # Test that objective and constraint respect fixed elements
    x0_full = free_to_full_proj(x0_free)
    
    # Black elements should be 1.0
    @test all(x0_full[black_cells] .== 1.0)
    # White elements should be 0.0
    @test all(x0_full[white_cells] .== 0.0)
    
    # Test gradient flow through projector
    grad_obj = Zygote.gradient(obj, x0_free)[1]
    @test length(grad_obj) == get_free_variable_count(free_to_full_proj)
    @test all(isfinite, grad_obj)
    
    # Run a few iterations of optimization using NLopt
    model = Nonconvex.Model(obj)
    Nonconvex.addvar!(model, zeros(get_free_variable_count(free_to_full_proj)), ones(get_free_variable_count(free_to_full_proj)), init=copy(x0_free))
    Nonconvex.add_ineq_constraint!(model, constr)
    
    # Solve with MMA algorithm
    options = MMAOptions(; maxiter=10, tol=Tolerance(; kkt = 1e-6))
    result = Nonconvex.optimize(model, MMA87(), x0_free, options=options)
    
    # Get optimized design
    x_opt_free = result.minimizer
    x_opt_full = free_to_full_proj(x_opt_free)
    
    # Verify constraints are still respected after optimization
    @test all(x_opt_full[black_cells] .== 1.0)
    @test all(x_opt_full[white_cells] .== 0.0)
    
    # Verify volume constraint is satisfied (within tolerance)
    v_final = constr(x_opt_free)
    @test abs(v_final) < 0.1  # Within 10% of target
    
    # Verify compliance decreased
    @test obj(x_opt_free) < obj(x0_free)
    
    # Verify gradients are zero at fixed elements (via projection)
    grad_final = Zygote.gradient(obj, x_opt_free)[1]
    @test all(isfinite, grad_final)
end
