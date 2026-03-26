using TopOpt, Test, LinearAlgebra, Ferrite
using TopOpt: matrix_free_apply_bcs!, MatrixFreeOperator, solve_matrix_free

@testset "Matrix Free BC Application" begin
    # Create simple system
    n = 10
    A = rand(n, n)
    A = A + A' + 10I  # Make symmetric positive definite
    
    b = rand(n)
    x = zeros(n)
    
    # Test with no boundary conditions
    A_copy = copy(A)
    b_copy = copy(b)
    matrix_free_apply_bcs!(A_copy, b_copy, x, Int[], Int[])
    @test A_copy ≈ A
    @test b_copy ≈ b
    
    # Test with boundary conditions
    fixed_dofs = [1, 3]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    A_mod = copy(A)
    b_mod = copy(b)
    x_bc = zeros(n)
    x_bc[fixed_dofs] = [1.0, 2.0]  # Prescribed values
    
    matrix_free_apply_bcs!(A_mod, b_mod, x_bc, fixed_dofs, free_dofs)
    
    # Check that fixed DOFs are handled correctly
    @test length(b_mod) == n
    @test size(A_mod) == (n, n)
end

@testset "MatrixFreeOperator Construction" begin
    # Create test matrix
    n = 20
    K = rand(n, n)
    K = K + K' + 100I
    
    fixed_dofs = [1, 5, 10]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    op = MatrixFreeOperator(K, free_dofs, fixed_dofs)
    
    @test op.K === K
    @test op.free_dofs == free_dofs
    @test op.fixed_dofs == fixed_dofs
    @test op.n_free == length(free_dofs)
end

@testset "MatrixFreeOperator Linear Map" begin
    # Create SPD matrix
    n = 15
    K_base = rand(n, n)
    K_base = K_base + K_base' + 100I
    
    fixed_dofs = [1, n]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    # Extract free-free block
    K_ff = K_base[free_dofs, free_dofs]
    
    op = MatrixFreeOperator(K_base, free_dofs, fixed_dofs)
    
    # Test matrix-vector multiplication
    x = rand(length(free_dofs))
    y = similar(x)
    
    mul!(y, op, x)
    y_expected = K_ff * x
    @test y ≈ y_expected
    
    # Test with different vectors
    x2 = ones(length(free_dofs))
    y2 = op * x2
    @test length(y2) == length(free_dofs)
end

@testset "MatrixFreeOperator Size" begin
    n = 20
    K = rand(n, n)
    fixed_dofs = [1, 2, 3]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    op = MatrixFreeOperator(K, free_dofs, fixed_dofs)
    
    @test size(op) == (length(free_dofs), length(free_dofs))
    @test size(op, 1) == length(free_dofs)
    @test size(op, 2) == length(free_dofs)
end

@testset "MatrixFreeOperator Properties" begin
    n = 12
    K = rand(n, n)
    K = K + K' + 50I
    
    fixed_dofs = [1, 6]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    op = MatrixFreeOperator(K, free_dofs, fixed_dofs)
    
    @test LinearAlgebra.issymmetric(op) == LinearAlgebra.issymmetric(K[free_dofs, free_dofs])
    @test op.n_free == length(free_dofs)
end

@testset "Matrix Free Solve" begin
    # Create simple SPD system
    n = 20
    K = rand(n, n)
    K = K + K' + 100I
    
    fixed_dofs = [1, n]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    # Known solution for free DOFs
    x_true = rand(length(free_dofs))
    f = K[free_dofs, free_dofs] * x_true
    
    # Extend to full system
    f_full = zeros(n)
    f_full[free_dofs] = f
    
    # Test solve
    x_result = solve_matrix_free(K, f_full, free_dofs, fixed_dofs)
    
    @test length(x_result) == n
    @test x_result[fixed_dofs] ≈ zeros(length(fixed_dofs))  # BC values
    
    # Check free DOFs solution (with tolerance for iterative solve)
    @test norm(x_result[free_dofs] - x_true) / norm(x_true) < 0.01
end

@testset "Matrix Free with Different Sizes" begin
    for n in [5, 10, 25]
        K = rand(n, n)
        K = K + K' + 50I
        
        n_fixed = max(1, div(n, 5))
        fixed_dofs = 1:n_fixed
        free_dofs = setdiff(1:n, fixed_dofs)
        
        op = MatrixFreeOperator(K, free_dofs, fixed_dofs)
        
        @test size(op, 1) == n - n_fixed
        @test size(op, 2) == n - n_fixed
        
        # Test multiplication
        x = rand(length(free_dofs))
        y = op * x
        @test length(y) == length(free_dofs)
    end
end

@testset "Matrix Free Edge Cases" begin
    # Single fixed DOF
    n = 10
    K = rand(n, n)
    K = K + K' + 100I
    
    fixed_dofs = [1]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    op = MatrixFreeOperator(K, free_dofs, fixed_dofs)
    @test size(op) == (n-1, n-1)
    
    # Many fixed DOFs
    fixed_dofs = 1:div(n, 2)
    free_dofs = setdiff(1:n, fixed_dofs)
    
    op2 = MatrixFreeOperator(K, free_dofs, fixed_dofs)
    @test size(op2, 1) == length(free_dofs)
end

@testset "Matrix Free Consistency" begin
    # Test that matrix-free gives same result as direct solve
    n = 15
    K = rand(n, n)
    K = K + K' + 100I
    
    fixed_dofs = [1, 3, 5]
    free_dofs = setdiff(1:n, fixed_dofs)
    
    # Create RHS
    f = rand(n)
    f[fixed_dofs] .= 0  # Zero prescribed displacements
    
    # Matrix-free solve
    x_mf = solve_matrix_free(K, f, free_dofs, fixed_dofs)
    
    # Direct solve for free DOFs
    K_ff = K[free_dofs, free_dofs]
    f_f = f[free_dofs]
    x_direct = K_ff \ f_f
    
    @test x_mf[free_dofs] ≈ x_direct
    @test x_mf[fixed_dofs] ≈ zeros(length(fixed_dofs))
end