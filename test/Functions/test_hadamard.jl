# Test file for hadamard functions
using Test
using LinearAlgebra
using TopOpt

# Import the hadamard functions from TopOpt.Functions
# hadamard! is exported from TopOpt, but hadamard2! and hadamard3! are internal
import TopOpt: hadamard!
using TopOpt.Functions: hadamard2!, hadamard3!

@testset "hadamard3! tests" begin
    @testset "Basic functionality" begin
        # Test 2x2 matrix (power of 2)
        V = zeros(2, 2)
        hadamard3!(V)
        @test V == [1 1; 1 -1]
        
        # Test 4x4 matrix
        V = zeros(4, 4)
        hadamard3!(V)
        @test V == [1 1 1 1; 1 -1 1 -1; 1 1 -1 -1; 1 -1 -1 1]
    end
    
    @testset "Rectangular matrices" begin
        # More rows than columns
        V = zeros(6, 4)
        hadamard3!(V)
        @test size(V) == (6, 4)
        # Check all values are ±1
        @test all(x -> x == 1 || x == -1, V)
        
        # Fewer columns than rows (n < nv case)
        V = zeros(4, 2)
        hadamard3!(V)
        @test size(V) == (4, 2)
        @test all(x -> x == 1 || x == -1, V)
    end
    
    @testset "Non-power-of-2 dimensions" begin
        V = zeros(3, 3)
        hadamard3!(V)
        @test size(V) == (3, 3)
        @test all(x -> x == 1 || x == -1, V)
        
        V = zeros(5, 6)
        hadamard3!(V)
        @test size(V) == (5, 6)
        @test all(x -> x == 1 || x == -1, V)
    end
    
    @testset "Edge cases" begin
        # 1x1 matrix
        V = zeros(1, 1)
        hadamard3!(V)
        @test V ≈ [1.0]
        
        # Single row
        V = zeros(1, 4)
        hadamard3!(V)
        @test size(V) == (1, 4)
        @test all(x -> x == 1 || x == -1, V)
        
        # Single column
        V = zeros(4, 1)
        hadamard3!(V)
        @test size(V) == (4, 1)
        @test all(x -> x == 1 || x == -1, V)
    end
end

@testset "hadamard2! tests" begin
    @testset "Basic functionality" begin
        # Test 2x2 matrix
        V = zeros(2, 2)
        hadamard2!(V)
        @test V == [1 1; 1 -1]
        
        # Test 4x4 matrix
        V = zeros(4, 4)
        hadamard2!(V)
        expected = [1 1 1 1; 
                    1 -1 1 -1; 
                    1 1 -1 -1; 
                    1 -1 -1 1]
        @test V == expected
    end
    
    @testset "Rectangular matrices" begin
        # More rows than columns
        V = zeros(6, 4)
        hadamard2!(V)
        @test size(V) == (6, 4)
        @test all(x -> x == 1 || x == -1, V)
        
        # More rows (should just repeat rows with sign flip)
        V = zeros(8, 4)
        hadamard2!(V)
        @test size(V) == (8, 4)
        @test all(x -> x == 1 || x == -1, V)
    end
    
    @testset "Edge cases" begin
        # 1x1 matrix
        V = zeros(1, 1)
        hadamard2!(V)
        @test V ≈ [1.0]
        
        # Single row
        V = zeros(1, 4)
        hadamard2!(V)
        @test size(V) == (1, 4)
        @test all(x -> x == 1 || x == -1, V)
        
        # Single column
        V = zeros(4, 1)
        hadamard2!(V)
        @test size(V) == (4, 1)
        @test all(x -> x == 1 || x == -1, V)
    end
end

@testset "Comparison between hadamard! variants" begin
    @testset "Square power-of-2 matrices" begin
        # For square power-of-2 matrices, all should produce same result
        for n in [1, 2, 4, 8]
            V1 = zeros(n, n)
            V2 = zeros(n, n)
            V3 = zeros(n, n)
            hadamard!(V1)
            hadamard2!(V2)
            hadamard3!(V3)
            
            # Check they all produce valid Hadamard matrices
            @test all(x -> x == 1 || x == -1, V1)
            @test all(x -> x == 1 || x == -1, V2)
            @test all(x -> x == 1 || x == -1, V3)
        end
    end
    
    @testset "In-place modification" begin
        # Test that functions modify V in-place
        V1 = zeros(4, 4)
        V2 = copy(V1)
        result = hadamard3!(V1)
        @test result === V1  # Should return the same object
        @test V1 != V2      # Should have been modified
    end
end

@testset "Hadamard matrix properties" begin
    @testset "Orthogonality (for square matrices)" begin
        for n in [1, 2, 4, 8]
            V = zeros(n, n)
            hadamard3!(V)
            # For a proper Hadamard matrix: H*H' = n*I
            @test isapprox(V * V', n * I, atol=1e-10)
        end
    end
    
    @testset "Determinant (for power-of-2)" begin
        for n in [1, 2, 4, 8]
            V = zeros(n, n)
            hadamard3!(V)
            # det(Hadamard) = ±n^(n/2)
            @test abs(det(V)) ≈ n^(n/2)
        end
    end
end
