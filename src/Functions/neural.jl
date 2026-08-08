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

struct NeuralNetwork{Tm,Ti1,Tp,Ti2,Tc} <: AbstractMLModel
    model::Tm
    init_params::Ti1
    params_to_out::Tp
    in_to_out::Ti2
    centroids::Tc
end

struct PredictFunction{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (pf::PredictFunction)(in::AbstractVector{<:Real})
    return PseudoDensities(pf.model.in_to_out(in))
end

struct TrainFunction{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (tf::TrainFunction)(p::AbstractVector{<:Real})
    return PseudoDensities(tf.model.params_to_out(p))
end

function (ml::NeuralNetwork)(x::AbstractVector{<:Coordinates})
    return PredictFunction(ml).(getfield.(x, :coords))
end
(ml::NeuralNetwork)(x::Coordinates) = PredictFunction(ml)(x.coords)
(ml::NeuralNetwork)(x::NNParams) = TrainFunction(ml)(x.p)
