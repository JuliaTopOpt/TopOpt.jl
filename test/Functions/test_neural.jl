using TopOpt
using Test
using Ferrite
using TopOpt.Functions: NeuralNetwork, Coordinates, NNParams, PredictFunction, TrainFunction, getcentroids

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
    
    @testset "NeuralNetwork problem constructor" begin
        # Create a simple problem
        problem = PointLoadCantilever(Val{:Linear}, (4, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
        
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))
        
        # Test NeuralNetwork constructor with problem and scale=true (default)
        ml = NeuralNetwork(nn, problem; scale=true)
        
        @test typeof(ml) <: TopOpt.Functions.AbstractMLModel
        @test typeof(ml.model) <: Flux.Chain
        @test length(ml.init_params) > 0
        @test length(ml.centroids) > 0
        @test typeof(ml.params_to_out) <: Function
        @test typeof(ml.in_to_out) <: Flux.Chain
        
        # Test with scale=false
        ml_noscale = NeuralNetwork(nn, problem; scale=false)
        @test typeof(ml_noscale) <: TopOpt.Functions.AbstractMLModel
        @test length(ml_noscale.centroids) == length(ml.centroids)
        
        # Test getcentroids function directly
        centroids = getcentroids(problem)
        @test length(centroids) > 0
        @test all(c -> length(c) == 2, centroids)  # 2D coordinates
    end
    
    @testset "PredictFunction and TrainFunction" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))
        
        # Create input coordinates
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        
        # Create NeuralNetwork
        ml = NeuralNetwork(nn, coords)
        
        # Test PredictFunction
        pred_fn = PredictFunction(ml)
        @test typeof(pred_fn) <: PredictFunction
        
        # Test calling PredictFunction with coordinates
        input_coords = [0.0, 0.0]
        result = pred_fn(input_coords)
        @test typeof(result) <: TopOpt.PseudoDensities
        
        # Test TrainFunction
        train_fn = TrainFunction(ml)
        @test typeof(train_fn) <: TrainFunction
        
        # Test calling TrainFunction with params
        params = ml.init_params
        result_train = train_fn(params)
        @test typeof(result_train) <: TopOpt.PseudoDensities
    end
    
    @testset "NeuralNetwork callable methods" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))
        
        # Create input coordinates
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        
        # Create NeuralNetwork
        ml = NeuralNetwork(nn, coords)
        
        # Test calling with Coordinates (line 49)
        coord = Coordinates([0.0, 0.0])
        result = ml(coord)
        @test typeof(result) <: TopOpt.PseudoDensities
        
        # Test calling with vector of Coordinates
        coords_vec = [Coordinates([0.0, 0.0]), Coordinates([0.5, 0.5])]
        results = ml(coords_vec)
        @test length(results) == 2
        @test all(r -> typeof(r) <: TopOpt.PseudoDensities, results)
        
        # Test calling with NNParams
        params = NNParams(ml.init_params)
        result_params = ml(params)
        @test typeof(result_params) <: TopOpt.PseudoDensities
    end
end
