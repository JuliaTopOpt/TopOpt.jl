module Functions

using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ..GPUUtils, CuArrays
using Parameters: @unpack
using TimerOutputs, CUDAnative, JuAFEM

export  Objective,
        Constraint,
        VolumeFunction,
        ComplianceFunction

const to = TimerOutput()

abstract type AbstractFunction{T} <: Function end

struct Objective{T, F <: AbstractFunction{T}} <: Function
    f::F
end

struct Constraint{T, F <: AbstractFunction{T}} <: Function
    f::F
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
(o::Constraint)(x, grad) = o.f(x, grad)

include("compliance.jl")
include("volume.jl")

end

