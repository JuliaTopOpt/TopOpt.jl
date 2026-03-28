using TopOpt
using TopOpt.TopOptProblems
using Test

@testset "Element stiffness matrix" begin

    # 1x1 linear quadrilateral cell
    # Isoparameteric
    # 2nd order quadrature rule

    E = 1
    nu = 0.3
    k = [
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
    ]
    Ke =
        E / (1 - nu^2) * [
            k[1] k[2] k[3] k[4] k[5] k[6] k[7] k[8]
            k[2] k[1] k[8] k[7] k[6] k[5] k[4] k[3]
            k[3] k[8] k[1] k[6] k[7] k[4] k[5] k[2]
            k[4] k[7] k[6] k[1] k[8] k[3] k[2] k[5]
            k[5] k[6] k[7] k[8] k[1] k[2] k[3] k[4]
            k[6] k[5] k[4] k[3] k[2] k[1] k[8] k[7]
            k[7] k[4] k[5] k[2] k[3] k[8] k[1] k[6]
            k[8] k[3] k[2] k[5] k[4] k[7] k[6] k[1]
        ]

    problem = HalfMBB(Val{:Linear}, (2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)

    #@test ElementFEAInfo(problem).Kes[1] ≈ Ke
end

@testset "ElementStiffnessMatrix interface" begin
    using TopOpt.Functions: ElementStiffnessMatrix

    # Create test matrices of different sizes
    test_matrices = [
        rand(3, 3),
        rand(8, 8),
        [1.0 2.0; 3.0 4.0],
        [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],
    ]

    @testset "length" begin
        for mat in test_matrices
            esm = ElementStiffnessMatrix(mat)
            @test length(esm) == length(mat)
            @test length(esm) == length(mat[:])
        end
    end

    @testset "size" begin
        for mat in test_matrices
            esm = ElementStiffnessMatrix(mat)
            # size without dimensions
            @test size(esm) == size(mat)
            # size with dimension
            @test size(esm, 1) == size(mat, 1)
            @test size(esm, 2) == size(mat, 2)
        end

        # Test with larger matrix
        big_mat = rand(10, 10)
        esm = ElementStiffnessMatrix(big_mat)
        @test size(esm) == (10, 10)
        @test size(esm, 1) == 10
        @test size(esm, 2) == 10
    end

    @testset "getindex" begin
        mat = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        esm = ElementStiffnessMatrix(mat)

        # Linear indexing
        @test esm[1] == mat[1]
        @test esm[5] == mat[5]
        @test esm[end] == mat[end]

        # Cartesian indexing
        @test esm[1, 1] == mat[1, 1]
        @test esm[2, 3] == mat[2, 3]
        @test esm[3, 2] == mat[3, 2]

        # Slicing
        @test esm[1:2, 1:2] == mat[1:2, 1:2]
        @test esm[:, 1] == mat[:, 1]
        @test esm[1, :] == mat[1, :]
    end

    @testset "multiplication *" begin
        mat = [1.0 2.0; 3.0 4.0]
        esm = ElementStiffnessMatrix(mat)

        # Scalar multiplication
        scalar = 2.5
        result = esm * scalar
        @test result isa ElementStiffnessMatrix
        @test result.Ke == mat * scalar

        # Integer scalar
        result2 = esm * 3
        @test result2 isa ElementStiffnessMatrix
        @test result2.Ke == mat * 3

        # Float scalar
        result3 = esm * 0.5
        @test result3 isa ElementStiffnessMatrix
        @test result3.Ke == mat * 0.5

        # Test with larger matrix
        big_mat = rand(4, 4)
        big_esm = ElementStiffnessMatrix(big_mat)
        result_big = big_esm * 10.0
        @test result_big isa ElementStiffnessMatrix
        @test result_big.Ke == big_mat * 10.0
    end

    @testset "AbstractMatrix interface" begin
        # Test that ElementStiffnessMatrix is an AbstractMatrix
        mat = rand(5, 5)
        esm = ElementStiffnessMatrix(mat)

        @test esm isa AbstractMatrix
        @test eltype(esm) == eltype(mat)

        # Test iteration (inherited from AbstractMatrix)
        @test collect(esm) == collect(mat)
    end
end
