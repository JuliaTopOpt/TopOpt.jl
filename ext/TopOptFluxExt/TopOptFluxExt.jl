module TopOptFluxExt

using Flux
using Flux: destructure
using TopOpt: TopOpt
using TopOpt.TopOptProblems: AbstractTopOptProblem
import TopOpt.Functions: NeuralNetwork
using TopOpt.Functions: getcentroids
using Statistics: mean, std

# The NeuralNetwork struct and all non-Flux methods live in the main package.
# This extension adds the two constructors that require Flux.destructure.

function NeuralNetwork(nn_model, input_coords::AbstractVector{<:AbstractVector{<:Real}})
    f = x -> nn_model(x)[1]
    @assert all(0 .<= f.(input_coords) .<= 1)
    p, re = destructure(nn_model)
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
            return (c .- m) ./ s
        end
    else
        scentroids = centroids
    end
    return NeuralNetwork(nn_model, scentroids)
end

end
