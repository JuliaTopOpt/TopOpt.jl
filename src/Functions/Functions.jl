module Functions

using ..TopOpt: whichdevice, CPU, GPU, TopOpt
using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra, Requires
using Parameters: @unpack
using TimerOutputs, JuAFEM, StaticArrays
using StatsFuns, MappedArrays, LazyArrays
using ..TopOptProblems: getdh
using SparseArrays

export  Objective,
        Constraint,
        VolumeFunction,
        ComplianceFunction,
        ZeroFunction,
        Sum,
        Product,
        BinPenaltyFunction,
        LinAggregation,
        QuadAggregation,
        QuadMaxAggregation,
        LinQuadAggregation,
        LinQuadMaxAggregation,
        AbstractFunction,
        getfevals,
        getmaxfevals,
        maxedfevals,
        getnvars,
        GlobalStress,
        project

const to = TimerOutput()

abstract type AbstractFunction{T} <: Function end
abstract type AbstractConstraint{T} <: AbstractFunction{T} end

@params struct Objective{T} <: AbstractFunction{T}
    f
end
Objective(::Type{T}, f) where {T} = Objective{T, typeof(f)}(f)
Objective(f::AbstractFunction{T}) where {T} = Objective(T, f)
Objective(f::Function) = Objective(Float64, f)

TopOpt.dim(o::Objective) = TopOpt.dim(o.f)

@inline function Base.getproperty(o::Objective, s::Symbol)
    s === :f && return getfield(o, :f)
    return getproperty(o.f, s)
end
@inline function Base.setproperty!(o::Objective, s::Symbol, v)
    s === :f && return setfield!(o, :f, v)
    return setproperty!(o.f, s, v)
end

@params struct BlockConstraint{T} <: AbstractConstraint{T}
    f
    s
    dim::Int
end
function BlockConstraint(::Type{T}, f, s, dim = dim(f)) where {T}
    return BlockConstraint{T, typeof(f), typeof(s)}(f, s, dim)
end
function BlockConstraint(f::AbstractFunction{T}, s, dim = dim(f)) where {T}
    return BlockConstraint(T, f, s, dim)
end
function BlockConstraint(f::Function, s, dim = dim(f))
    return BlockConstraint(Float64, f, s, dim)
end
TopOpt.dim(c::BlockConstraint) = c.dim

@params struct Constraint{T} <: AbstractConstraint{T}
    f
    s
end
Constraint(::Type{T}, f, s) where {T} = Constraint{T, typeof(f), typeof(s)}(f, s)
Constraint(f::AbstractFunction{T}, s) where {T} = Constraint(T, f, s)
Constraint(f::Function, s) = Constraint(Float64, f, s)
TopOpt.dim(c::Constraint) = 1

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
Utilities.getsolver(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver
Utilities.getpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getpenalty
Utilities.setpenalty!(o::Union{Objective, Constraint}, p) = setpenalty!(getsolver(getfunction(o)), p)
Utilities.getprevpenalty(o::Union{Objective, Constraint}) = o |> getfunction |> getsolver |> getprevpenalty

(o::Objective)(args...) = o.f(args...)

(c::Constraint)(args...) = c.f(args...) - c.s

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
function (z::ZeroFunction)(x, g=nothing)
    z.fevals += 1
    if g !== nothing
    g .= 0
    end
    return zero(eltype(g))
end

getmaxfevals(::ZeroFunction) = Inf
maxedfevals(::ZeroFunction) = false
@inline function Base.getproperty(z::ZeroFunction{T}, f::Symbol) where {T}
    f === :reuse && return false
    f === :grad && return zero(T)
    return getfield(z, f)
end

include("compliance.jl")
include("volume.jl")
include("stress.jl")
include("integrality_penalty.jl")
include("sum.jl")
include("product.jl")
include("lin_quad_aggregation.jl")

end
