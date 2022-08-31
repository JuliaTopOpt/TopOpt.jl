function getcentroids(problem::AbstractTopOptProblem)
    dh = problem.ch.dh
    return map(CellIterator(dh)) do cell
        return Vector(mean(cell.coords))
    end
end

abstract type AbstractMLModel end

# in -- params --> out -> filter -> compliance === loss
# params -- in --> out -> filter -> compliance === loss

@params struct NeuralNetwork <: AbstractMLModel
    model::Any
    init_params::Any
    params_to_out::Any
    in_to_out::Any
end
function NeuralNetwork(nn_model, input_coords::AbstractVector)
    f = x -> nn_model(x)[1]
    @assert all(0 .<= f.(input_coords) .<= 1)
    p, re = Flux.destructure(nn_model)
    return NeuralNetwork(
        nn_model, Float64.(p), p -> getindex.(re(p).(input_coords), 1), nn_model
    )
end
function NeuralNetwork(nn_model, problem::AbstractTopOptProblem)
    centroids = getcentroids(problem)
    m, s = mean(centroids), std(centroids)
    scentroids = map(centroids) do c
        (c .- m) ./ s
    end
    return NeuralNetwork(nn_model, scentroids)
end

@params struct PredictFunction <: Function
    model::AbstractMLModel
end
function (pf::PredictFunction)(in)
    return PseudoDensities(pf.model.in_to_out(in))
end

@params struct TrainFunction <: Function
    model::AbstractMLModel
end
function (tf::TrainFunction)(p)
    return PseudoDensities(tf.model.params_to_out(p))
end
