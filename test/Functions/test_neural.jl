using TopOpt
using Test
using Ferrite

@testset "Neural Network Functions" begin
    @testset "NeuralNetwork Construction" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 5, relu), Flux.Dense(5, 1, sigmoid))
        
        # Create input coordinates (simulating centroids)
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        
        # Test NeuralNetwork constructor with coordinates
        ml = NeuralNetwork(nn, coords)
        
        @test typeof(ml) <: TopOpt.Functions.AbstractMLModel
        @test typeof(ml.model) <: Flux.Chain
        @test length(ml.init_params) > 0
        @test length(ml.centroids) == 3
        @test typeof(ml.params_to_out) <: Function
        @test typeof(ml.in_to_out) <: Flux.Chain
    end
end