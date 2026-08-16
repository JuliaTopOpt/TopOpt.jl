using TopOpt
using Test
using Ferrite
using Flux
using TopOpt:
    NeuralNetworkFun,
    Coordinates,
    NNParams,
    PredictFunctionFun,
    TrainFunctionFun,
    getcentroids,
    AbstractMLModel

@testset "Neural Network Functions" begin
    @testset "NeuralNetworkFun Construction" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 5, relu), Flux.Dense(5, 1, sigmoid))

        # Create input coordinates (simulating centroids)
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

        # Test NeuralNetworkFun constructor with coordinates
        ml = NeuralNetworkFun(nn, coords)

        @test typeof(ml) <: AbstractMLModel
        @test typeof(ml.model) <: Flux.Chain
        @test length(ml.init_params) > 0
        @test length(ml.centroids) == 3
        @test typeof(ml.params_to_out) <: Function
        @test typeof(ml.in_to_out) <: Flux.Chain
    end

    @testset "NeuralNetworkFun problem constructor" begin
        # Create a simple problem
        problem = PointLoadCantilever((4, 4), (1.0, 1.0), 1.0, 0.3, 1.0)

        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))

        # Test NeuralNetworkFun constructor with problem and scale=true (default)
        ml = NeuralNetworkFun(nn, problem; scale=true)

        @test typeof(ml) <: AbstractMLModel
        @test typeof(ml.model) <: Flux.Chain
        @test length(ml.init_params) > 0
        @test length(ml.centroids) > 0
        @test typeof(ml.params_to_out) <: Function
        @test typeof(ml.in_to_out) <: Flux.Chain

        # Test with scale=false
        ml_noscale = NeuralNetworkFun(nn, problem; scale=false)
        @test typeof(ml_noscale) <: AbstractMLModel
        @test length(ml_noscale.centroids) == length(ml.centroids)

        # Test getcentroids function directly
        centroids = getcentroids(problem)
        @test length(centroids) > 0
        @test all(c -> length(c) == 2, centroids)  # 2D coordinates
    end

    @testset "PredictFunctionFun and TrainFunctionFun" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))

        # Create input coordinates
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

        # Create NeuralNetworkFun
        ml = NeuralNetworkFun(nn, coords)

        # Test PredictFunctionFun
        pred_fn = PredictFunctionFun(ml)
        @test typeof(pred_fn) <: PredictFunctionFun

        # Test calling PredictFunctionFun with coordinates
        input_coords = [0.0, 0.0]
        result = pred_fn(input_coords)
        @test typeof(result) <: PseudoDensities

        # Test TrainFunctionFun
        train_fn = TrainFunctionFun(ml)
        @test typeof(train_fn) <: TrainFunctionFun

        # Test calling TrainFunctionFun with params
        params = ml.init_params
        result_train = train_fn(params)
        @test typeof(result_train) <: PseudoDensities
    end

    @testset "NeuralNetworkFun callable methods" begin
        # Create a simple NN model
        nn = Flux.Chain(Flux.Dense(2, 3, relu), Flux.Dense(3, 1, sigmoid))

        # Create input coordinates
        coords = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

        # Create NeuralNetworkFun
        ml = NeuralNetworkFun(nn, coords)

        # Test calling with Coordinates (line 49)
        coord = Coordinates([0.0, 0.0])
        result = ml(coord)
        @test typeof(result) <: PseudoDensities

        # Test calling with vector of Coordinates
        coords_vec = [Coordinates([0.0, 0.0]), Coordinates([0.5, 0.5])]
        results = ml(coords_vec)
        @test length(results) == 2
        @test all(r -> typeof(r) <: PseudoDensities, results)

        # Test calling with NNParams
        params = NNParams(ml.init_params)
        result_params = ml(params)
        @test typeof(result_params) <: PseudoDensities
    end
end
