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
    model
    init_params
    params_to_out
    in_to_out
end
function NeuralNetwork(nn_model, input_coords::AbstractVector)
    f = x -> nn_model(x)[1]
    @assert all(0 .<= f.(input_coords) .<= 1)
    p, re = Flux.destructure(nn_model)
    return NeuralNetwork(nn_model, p, p -> getindex.(re(p).(input_coords), 1), nn_model)
end
function NeuralNetwork(nn_model, problem::AbstractTopOptProblem)
    return NeuralNetwork(nn_model, getcentroids(problem))
end

@params struct PredictFunction <: Function
    model::AbstractMLModel
end
function (pf::PredictFunction)(in)
    return pf.model.in_to_out(in)
end

@params struct TrainFunction <: Function
    model::AbstractMLModel
end
function (tf::TrainFunction)(p)
    return tf.model.params_to_out(p)
end
