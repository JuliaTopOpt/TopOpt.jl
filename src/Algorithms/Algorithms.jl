module Algorithms

using ..MMA, ..Functions, Optim
using TimerOutputs, Setfield, StaticArrays
using Parameters: @unpack, @pack!
using ..GPUUtils, ..Utilities

export  MMAOptimizer

const to = TimerOutput()

abstract type TopOptAlgorithm end

# Solid isotropic material with penalisation
include(joinpath("SIMP", "SIMP.jl"))

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
