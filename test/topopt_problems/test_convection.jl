using TopOpt
using TopOpt.TopOptProblems
using Test
using LinearAlgebra
using SparseArrays

@testset "Thermal Convection Matrix Assembly" begin
    # Creates a tiny 2x2 dummy grid
    nels = (2, 2)
    sizes = (1.0, 1.0)
    k = 1.0

    # Applying convection to the top edge
    convection = Dict{String,Tuple{Float64,Float64}}("top" => (10.0, 20.0))
    problem = TopOpt.TopOptProblems.HeatConductionProblem(
        Val{:Linear}, nels, sizes, k; convection=convection
    )

    # Building the matrix using explicit path to avoid export issues
    K_conv = TopOpt.TopOptProblems.assemble_convection_matrix(problem)

    # --- THE TESTS ---
    #  A 2x2 linear grid has 3x3 = 9 nodes. 1 DOF per node = 9x9 matrix.
    @test size(K_conv) == (9, 9)
    # It must be sparse for performance
    @test issparse(K_conv)
    # The math dictates it must be perfectly symmetric
    @test issymmetric(K_conv)
    # Because we applied convection, there should be non-zero elements
    @test nnz(K_conv) > 0
    # Check the fallback (Empty convection should return an empty sparse matrix)
    prob_empty = TopOpt.TopOptProblems.HeatConductionProblem(Val{:Linear}, nels, sizes, k)
    K_empty = TopOpt.TopOptProblems.assemble_convection_matrix(prob_empty)
    @test nnz(K_empty) == 0
end
