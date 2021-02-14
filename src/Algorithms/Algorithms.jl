module Algorithms

using Nonconvex, ..Functions, Optim, Parameters, Requires
using Nonconvex: AbstractModel
using ..TopOpt: whichdevice, AbstractDevice, CPU, GPU, PENALTY_BEFORE_INTERPOLATION
using TimerOutputs, Setfield, StaticArrays
using Parameters: @unpack, @pack!
using ..Utilities, JuAFEM
using LinearAlgebra

export  Optimizer,
        SIMP,
        ExponentialContinuation,
        ContinuationSIMP,
        AdaptiveSIMP,
        MMAOptionsGen,
        CSIMPOptions,
        BESO,
        GESO,
        Continuation,
        PowerContinuation

const to = TimerOutput()

abstract type TopOptAlgorithm end

# Solid isotropic material with penalisation
include(joinpath("SIMP", "SIMP.jl"))

# Bidirectional evolutionary strctural optimisation
include("beso.jl")

# Genetic evolutionary structural optimisation
include("geso.jl")

end
