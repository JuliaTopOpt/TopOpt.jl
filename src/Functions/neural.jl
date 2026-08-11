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

"""
    NeuralNetwork(nn, problem)

Re-parametrizes the design in terms of a neural network's weights and biases.
`nn` is a `Flux.jl` model whose first layer takes 2 (or 3) coordinates for 2D
(or 3D) and whose last layer returns a scalar in [0, 1]. In prediction mode
the network is called on each element's centroid to produce that element's
design variable.
"""
struct NeuralNetwork{Tm,Ti1,Tp,Ti2,Tc} <: AbstractMLModel
    model::Tm
    init_params::Ti1
    params_to_out::Tp
    in_to_out::Ti2
    centroids::Tc
end

"""
    PredictFunction(nn_model)

Prediction function that applies the neural network to each element centroid
to produce the design variable. Used for evaluating a trained model.
"""
struct PredictFunction{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (pf::PredictFunction)(in::AbstractVector{<:Real})
    return PseudoDensities(pf.model.in_to_out(in))
end

"""
    TrainFunction(nn_model)

The training function used in the re-parameterized topology optimization
formulation. Takes the vector of neural-network weights/biases `p` and returns
the vector of element-wise design variables `x`.
"""
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
