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

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
