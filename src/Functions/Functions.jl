module Functions

using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ..GPUUtils, CuArrays
using Parameters: @unpack
using TimerOutputs, CUDAnative

export  AbstractObjective,
        AbstractConstraint,
        VolConstr,
        ComplianceObj

const to = TimerOutput()

abstract type AbstractObjective{T} <: Function end
abstract type AbstractConstraint <: Function end

include("compliance.jl")
include("constraints.jl")

end

