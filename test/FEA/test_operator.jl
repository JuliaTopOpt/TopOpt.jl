using TopOpt, LinearAlgebra, Test
using Ferrite: ndofs, apply_zero!
using TopOpt.TopOptProblems: getncells
using TopOpt.FEA: ElementFEAInfo, GlobalFEAInfo, assemble!

# Create a simple mesh and problem for testing
nels = (2, 2)
size = (1.0, 1.0)
E = 1.0
ν = 0.3
f = 1.0

# Create problem
problem = HalfMBB(Val{:Linear}, nels, size, E, ν, f)
n_dofs = ndofs(problem.ch.dh)

# Create element info and global info needed for operators
elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
globalinfo = GlobalFEAInfo(problem)

# Design variables (density vector)
x = fill(1.0, getncells(problem.ch.dh.grid))

# Create a test vector
b = rand(n_dofs)
apply_zero!(b, problem.ch)

@testset "MatrixFreeOperator * operator" begin
    # Build matrix-free operator with required parameters
    T = Float64
    meandiag = 1.0  # Simplified mean diagonal
    xes = deepcopy(elementinfo.fes)
    fixed_dofs = problem.ch.prescribed_dofs
    free_dofs = setdiff(1:n_dofs, fixed_dofs)
    xmin = 0.001
    penalty = PowerPenalty{T}(3.0)
    conv = DefaultCriteria()

    # Get force vector from globalinfo
    f_vec = globalinfo.f

    mfree = MatrixFreeOperator(
        f_vec, elementinfo, meandiag, x, xes,
        fixed_dofs, free_dofs, xmin, penalty, conv
    )

    # Test * operator
    result = mfree * b
    @test result isa Vector
    @test length(result) == n_dofs

    # Test that the result is not NaN or Inf
    @test all(isfinite, result)

    # Test mul! function
    result2 = similar(result)
    mul!(result2, mfree, b)
    @test result2 ≈ result
end

@testset "MatrixOperator * operator" begin
    # Assemble the global stiffness matrix
    assemble!(globalinfo, problem, elementinfo, x, PowerPenalty{Float64}(3.0), 0.001)

    # Get the assembled matrix
    K = globalinfo.K

    # Create MatrixOperator with force vector and convergence criteria
    f_vec = globalinfo.f
    conv = DefaultCriteria()

    mop = MatrixOperator(K.data, f_vec, conv)

    # Test * operator
    result = mop * b
    @test result isa Vector
    @test length(result) == n_dofs

    # Test that the result is not NaN or Inf
    @test all(isfinite, result)

    # Test mul! function
    result2 = similar(result)
    mul!(result2, mop, b)
    @test result2 ≈ result
end