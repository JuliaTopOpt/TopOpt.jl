module Functions

using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ..GPUUtils, CuArrays
using ForwardDiff, LinearAlgebra, GPUArrays, StaticArrays
using Parameters: @unpack
import CUDAdrv
using TimerOutputs, CUDAnative, JuAFEM
using StatsFuns, MappedArrays, LazyArrays
using ..TopOptProblems: getdh

export  Objective,
        Constraint,
        VolumeFunction,
        ComplianceFunction,
        ZeroFunction,
        AbstractFunction,
        getfevals,
        getmaxfevals,
        maxedfevals,
        getnvars,
        GlobalStress,
        project

const to = TimerOutput()
const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

abstract type AbstractFunction{T} <: Function end

@params struct Objective <: Function
    f
end
@inline function Base.getproperty(o::Objective, s::Symbol)
    s === :f && return getfield(o, :f)
    return getproperty(o.f, s)
end
@inline function Base.setproperty!(o::Objective, s::Symbol, v)
    s === :f && return setfield!(o, :f, v)
    return setproperty!(o.f, s, v)
end

@params struct Constraint <: Function
    f
    s
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

# For feasibility problems
mutable struct ZeroFunction{T, Tsolver} <: AbstractFunction{T}
    solver::Tsolver
    fevals::Int
end
function ZeroFunction(solver::AbstractFEASolver)
    return ZeroFunction{eltype(solver.vars), typeof(solver)}(solver, 0)
end
function (z::ZeroFunction)(x, g) 
    z.fevals += 1
    g .= 0
    return zero(eltype(g))
end

getmaxfevals(::ZeroFunction) = Inf
maxedfevals(::ZeroFunction) = false
@inline function Base.getproperty(z::ZeroFunction, f::Symbol)
    f === :reuse && return false
    return getfield(z, f)
end
GPUUtils.whichdevice(z::ZeroFunction) = whichdevice(z.solver)
GPUUtils.CuArrays.cu(::ZeroFunction{T}) where T = ZeroFunction{T}()

include("compliance.jl")
include("volume.jl")
include("stress.jl")

end
