module Algorithms

using Nonconvex, ..Functions, Parameters
using ..TopOpt: PseudoDensities
using Setfield, StaticArrays
using Parameters: @unpack, @pack!
using ..Utilities, Ferrite
using LinearAlgebra, Zygote

export BESO, GESO

abstract type TopOptAlgorithm end

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
