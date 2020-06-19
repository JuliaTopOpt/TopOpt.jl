module Functions

using ..TopOpt: dim, whichdevice, CPU, GPU, jtvp!, TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra, Requires
using Parameters: @unpack
using TimerOutputs, JuAFEM, StaticArrays
using StatsFuns, MappedArrays, LazyArrays
using ..TopOptProblems: getdh
using SparseArrays, Tracker, Statistics

export  Objective,
        Constraint,
        BlockConstraint,
        Volume,
        Compliance,
        MeanCompliance,
        BlockCompliance,
        MeanVar,
        MeanStd,
        ScalarValued,
        Zero,
        Sum,
        Product,
        Log,
        BinPenalty,
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
        project,
        generate_scenarios,
        hutch_rand!,
        hadamard!

const to = TimerOutput()

abstract type AbstractFunction{T} <: Function end
# Fallback for scalar-valued functions
"""
    jtvp!(out, f, x, v; runf = true)

Finds the product `J'v` and writes it to `out`, where `J` is the Jacobian of `f` at `x`. If `runf` is `true`, the function `f` will be run, otherwise the function will be assumed to have been run by the caller.
"""
function TopOpt.jtvp!(out, f::AbstractFunction, x, v; runf = true)
    runf && f(x)
    @assert length(v) == 1
    @assert all(isfinite, f.grad)
    @assert all(isfinite, v)
    out .= f.grad .* v
    return out
end

abstract type AbstractConstraint{T} <: AbstractFunction{T} end

@params struct Objective{T} <: AbstractFunction{T}
    f
end
Objective(::Type{T}, f) where {T <: Real} = Objective{T, typeof(f)}(f)
Objective(f::AbstractFunction{T}) where {T <: Real} = Objective(T, f)
Objective(f::Function) = Objective(Float64, f)
@forward_property Objective f

TopOpt.dim(o::Objective) = TopOpt.dim(o.f)
TopOpt.getnvars(o::Objective) = length(o.grad)

@params struct Constraint{T} <: AbstractConstraint{T}
    f
    s
end
Constraint(::Type{T}, f, s) where {T} = Constraint{T, typeof(f), typeof(s)}(f, s)
Constraint(f::AbstractFunction{T}, s) where {T} = Constraint(T, f, s)
Constraint(f::Function, s) = Constraint(Float64, f, s)
@forward_property Constraint f
TopOpt.dim(c::Constraint) = 1

@params struct BlockConstraint{T} <: AbstractConstraint{T}
    f
    s::Union{T, AbstractVector{T}}
    dim::Int
end
function BlockConstraint(::Type{T}, f, s, dim = dim(f)) where {T}
    return BlockConstraint(f, convert.(T, s), dim)
end
function BlockConstraint(f::AbstractFunction{T}, s::Union{Any, AbstractVector}) where {T}
    return BlockConstraint(f, convert.(T, s), dim(f))
end
function BlockConstraint(f::Function, s::Union{Any, AbstractVector})
    return BlockConstraint(f, s, dim(f))
end
@forward_property BlockConstraint f
TopOpt.dim(c::BlockConstraint) = c.dim
TopOpt.getnvars(c::BlockConstraint) = TopOpt.getnvars(c.f)
(bc::BlockConstraint)(x) = bc.f(x) .- bc.s

TopOpt.jtvp!(out, f::BlockConstraint, x, v; runf=true) = jtvp!(out, f.f, x, v, runf=runf)

Base.broadcastable(o::Union{Objective, AbstractConstraint}) = Ref(o)
getfunction(o::Union{Objective, AbstractConstraint}) = o.f
getfunction(f::AbstractFunction) = f
Utilities.getsolver(o::Union{Objective, AbstractConstraint}) = o |> getfunction |> getsolver
Utilities.getpenalty(o::Union{Objective, AbstractConstraint}) = o |> getfunction |> getsolver |> getpenalty
Utilities.setpenalty!(o::Union{Objective, AbstractConstraint}, p) = setpenalty!(getsolver(getfunction(o)), p)
Utilities.getprevpenalty(o::Union{Objective, AbstractConstraint}) = o |> getfunction |> getsolver |> getprevpenalty

(o::Objective)(args...) = o.f(args...)

(c::Constraint)(args...) = c.f(args...) - c.s

getfevals(o::Union{AbstractConstraint, Objective}) = o |> getfunction |> getfevals
getfevals(f::AbstractFunction) = f.fevals
getmaxfevals(o::Union{AbstractConstraint, Objective}) = o |> getfunction |> getmaxfevals
getmaxfevals(f::AbstractFunction) = f.maxfevals
maxedfevals(o::Union{Objective, AbstractConstraint}) = maxedfevals(o.f)
maxedfevals(f::AbstractFunction) = getfevals(f) >= getmaxfevals(f)

# For feasibility problems
mutable struct Zero{T, Tsolver} <: AbstractFunction{T}
    solver::Tsolver
    fevals::Int
end
function Zero(solver::AbstractFEASolver)
    return Zero{eltype(solver.vars), typeof(solver)}(solver, 0)
end
function (z::Zero)(x, g=nothing)
    z.fevals += 1
    if g !== nothing
        g .= 0
    end
    return zero(eltype(g))
end

getmaxfevals(::Zero) = Inf
maxedfevals(::Zero) = false
@inline function Base.getproperty(z::Zero{T}, f::Symbol) where {T}
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
include("log.jl")
include("lin_quad_aggregation.jl")
include("trace.jl")
include("mean_compliance.jl")
include("block_compliance.jl")
include("mean_var_std.jl")

end
