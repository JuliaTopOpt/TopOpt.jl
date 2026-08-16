"""
    Coordinates(coords)

Wrapper for the centroid coordinates of the elements, used as the input to the
neural-network model in [`NeuralNetworkFun`](@ref).
"""
struct Coordinates{C}
    coords::C
end

"""
    NNParams(p)

Wrapper for the neural network's weights and biases `p`, the design variables
of a neural-network-parametrized topology optimization.
"""
struct NNParams{W}
    p::W
end

"""
    getcentroids(problem::AbstractTopOptProblem)

Return a vector of the element centroid coordinates for `problem`.
"""
function getcentroids(problem::AbstractTopOptProblem)
    dh = problem.ch.dh
    return map(CellIterator(dh)) do cell
        return Vector(mean(cell.coords))
    end
end

"""
    AbstractMLModel

Abstract supertype for machine-learning models that re-parametrize the design
(e.g. [`NeuralNetworkFun`](@ref)).
"""
abstract type AbstractMLModel end

"""
    NeuralNetworkFun(nn, problem)

Re-parametrizes the design in terms of a neural network's weights and biases.
`nn` is a `Flux.jl` model whose first layer takes 2 (or 3) coordinates for 2D
(or 3D) and whose last layer returns a scalar in [0, 1]. In prediction mode
the network is called on each element's centroid to produce that element's
design variable.
"""
struct NeuralNetworkFun{Tm,Ti1,Tp,Ti2,Tc} <: AbstractMLModel
    model::Tm
    init_params::Ti1
    params_to_out::Tp
    in_to_out::Ti2
    centroids::Tc
end

"""
    PredictFunctionFun(nn_model)

Prediction function that applies the neural network to each element centroid
to produce the design variable. Used for evaluating a trained model.
"""
struct PredictFunctionFun{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (pf::PredictFunctionFun)(in::AbstractVector{<:Real})
    return PseudoDensities(pf.model.in_to_out(in))
end

"""
    TrainFunctionFun(nn_model)

The training function used in the re-parameterized topology optimization
formulation. Takes the vector of neural-network weights/biases `p` and returns
the vector of element-wise design variables `x`.
"""
struct TrainFunctionFun{Tm<:AbstractMLModel} <: Function
    model::Tm
end
function (tf::TrainFunctionFun)(p::AbstractVector{<:Real})
    return PseudoDensities(tf.model.params_to_out(p))
end

function (ml::NeuralNetworkFun)(x::AbstractVector{<:Coordinates})
    return PredictFunctionFun(ml).(getfield.(x, :coords))
end
(ml::NeuralNetworkFun)(x::Coordinates) = PredictFunctionFun(ml)(x.coords)
(ml::NeuralNetworkFun)(x::NNParams) = TrainFunctionFun(ml)(x.p)
