struct Coordinates{C}
    coords::C
end
struct NNParams{W}
    p::W
end

function getcentroids(problem::AbstractTopOptProblem)
    dh = problem.ch.dh
    return map(CellIterator(dh)) do cell
        return Vector(mean(cell.coords))
    end
end

abstract type AbstractMLModel end

# in -- params --> out -> filter -> compliance === loss
# params -- in --> out -> filter -> compliance === loss

struct NeuralNetwork{Tm,Ti1,Tp,Ti2,Tc} <: AbstractMLModel
    model::Tm
    init_params::Ti1
    params_to_out::Tp
    in_to_out::Ti2
    centroids::Tc
end
function NeuralNetwork(nn_model, input_coords::AbstractVector)
    f = x -> nn_model(x)[1]
    @assert all(0 .<= f.(input_coords) .<= 1)
    p, re = Flux.destructure(nn_model)
    return NeuralNetwork(
        nn_model,
        Float64.(p),
        p -> getindex.(re(p).(input_coords), 1),
        nn_model,
        input_coords,
    )
end
function NeuralNetwork(nn_model, problem::AbstractTopOptProblem; scale=true)
    centroids = getcentroids(problem)
    if scale
        m, s = mean(centroids), std(centroids)
        scentroids = map(centroids) do c
            (c .- m) ./ s
        end
    else
        scentroids = centroids
    end
    return NeuralNetwork(nn_model, scentroids)
end

struct PredictFunction{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (pf::PredictFunction)(in)
    return PseudoDensities(pf.model.in_to_out(in))
end

struct TrainFunction{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (tf::TrainFunction)(p)
    return PseudoDensities(tf.model.params_to_out(p))
end

function (ml::NeuralNetwork)(x::AbstractVector{<:Coordinates})
    return PredictFunction(ml).(x)
end
(ml::NeuralNetwork)(x::Coordinates) = PredictFunction(ml)(x.coords)
(ml::NeuralNetwork)(x::NNParams) = TrainFunction(ml)(x.p)
