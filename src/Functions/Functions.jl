module Functions

using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ..GPUUtils, CuArrays
using ForwardDiff, LinearAlgebra, GPUArrays
using Parameters: @unpack
import CUDAdrv
using TimerOutputs, CUDAnative, JuAFEM

export  Objective,
        Constraint,
        VolumeFunction,
        ComplianceFunction,
        AbstractFunction

const to = TimerOutput()
const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

abstract type AbstractFunction{T} <: Function end

struct Objective{F} <: Function
    f::F
end

struct Constraint{F, S} <: Function
    f::F
    s::S
end

getfunction(o::Union{Objective, Constraint}) = o.f
GPUUtils.whichdevice(o::Union{Objective, Constraint}) = o |> getfunction |> whichdevice
Utilities.getsolver(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver
Utilities.getpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getpenalty
Utilities.setpenalty!(o::Union{Objective, Constraint}, p) = setpenalty!(getsolver(getfunction(o)), p)
Utilities.getprevpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getprevpenalty

@define_cu(Objective, :f)
(o::Objective)(x, grad) = o.f(x, grad)

@define_cu(Constraint, :f)
(c::Constraint)(x, grad) = c.f(x, grad) - c.s

include("compliance.jl")
include("volume.jl")

end
