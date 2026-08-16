module Algorithms

using Nonconvex, ..Functions, Parameters
using ..TopOpt: PseudoDensities
using ..CheqFilters: AbstractCheqFilter
using Parameters: @unpack, @pack!
using ..Utilities
using LinearAlgebra: dot
using Random: rand, seed!
using Zygote: pullback
using Ferrite: getncells
using StaticArrays: MVector
using Ferrite
using Zygote
using Random

export BESO, GESO, TopOptAlgorithm

"""
    TopOptAlgorithm

Abstract base type for topology-optimization-specific algorithms
(`BESO`, `GESO`). General-purpose nonlinear optimizers (MMA, IPOPT, TOBS) are
provided through `Nonconvex.jl` and are not subtypes of this abstract type.
"""
abstract type TopOptAlgorithm end

# Set black elements to solid (`1`) and white elements to void (`0`) in both
# the design vector `vars` and its rounded `topology`, leaving all other
# elements untouched. Shared by `BESO` and `GESO`.
function initialize_black_white!(topology, vars, black, white)
    T = eltype(topology)
    @inbounds for i in eachindex(topology)
        if !isempty(black) && black[i]
            topology[i] = T(1)
            vars[i] = T(1)
        elseif !isempty(white) && white[i]
            topology[i] = T(0)
            vars[i] = T(0)
        end
    end
    return nothing
end

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
