module Algorithms

using ..MMA, ..Functions
using TimerOutputs, Setfield, StaticArrays
using Parameters: @unpack, @pack!

const to = TimerOutput()

abstract type TopOptAlgorithm end

# Solid isotropic material with penalisation
include(joinpath("SIMP", "SIMP.jl"))

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
