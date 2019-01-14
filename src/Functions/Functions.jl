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
        AbstractFunction,
        getfevals,
        getmaxfevals,
        maxedfevals,
        getnvars

const to = TimerOutput()
const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

abstract type AbstractFunction{T} <: Function end

struct Objective{F} <: Function
    f::F
end
@inline function Base.getproperty(o::Objective, s::Symbol)
    s === :f && return getfield(o, :f)
    return getproperty(o.f, s)
end
@inline function Base.setproperty!(o::Objective, s::Symbol, v)
    s === :f && return setfield!(o, :f, v)
    return setproperty!(o.f, s, v)
end

struct Constraint{F, S} <: Function
    f::F
    s::S
end
@inline function Base.getproperty(c::Constraint, s::Symbol)
    s === :f && return getfield(c, :f)
    s === :s && return getfield(c, :s)
    return getproperty(c.f, s)
end
@inline function Base.setproperty!(c::Constraint, s::Symbol, v)
    s === :f && return setfield!(c, :f, v)
    s === :s && return setfield!(c, :s, v)
    s === :reuse && return setproperty!(c.f, :reuse, v)
    return setfield!(c.f, s, v)
end

Base.broadcastable(o::Union{Objective, Constraint}) = Ref(o)
getfunction(o::Union{Objective, Constraint}) = o.f
getfunction(f::AbstractFunction) = f
GPUUtils.whichdevice(o::Union{Objective, Constraint}) = o |> getfunction |> whichdevice
Utilities.getsolver(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver
Utilities.getpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getpenalty
Utilities.setpenalty!(o::Union{Objective, Constraint}, p) = setpenalty!(getsolver(getfunction(o)), p)
Utilities.getprevpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getprevpenalty

@define_cu(Objective, :f)
(o::Objective)(x, grad) = o.f(x, grad)

@define_cu(Constraint, :f)
(c::Constraint)(x, grad) = c.f(x, grad) - c.s

getfevals(o::Union{Constraint, Objective}) = o |> getfunction |> getfevals
getfevals(f::AbstractFunction) = f.fevals
getmaxfevals(o::Union{Constraint, Objective}) = o |> getfunction |> getmaxfevals
getmaxfevals(f::AbstractFunction) = f.maxfevals
maxedfevals(o::Union{Objective, Constraint}) = maxedfevals(o.f)
maxedfevals(f::AbstractFunction) = getfevals(f) >= getmaxfevals(f)
getnvars(o::Union{Constraint, Objective}) = o |> getfunction |> getnvars
getnvars(f::AbstractFunction) = length(f.solver.vars)

include("compliance.jl")
include("volume.jl")

end
