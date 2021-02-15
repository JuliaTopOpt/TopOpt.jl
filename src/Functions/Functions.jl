module Functions

using ..TopOpt: dim, whichdevice, CPU, GPU, jtvp!, TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra, Requires
using Parameters: @unpack
using TimerOutputs, JuAFEM, StaticArrays
using StatsFuns, MappedArrays, LazyArrays
using ..TopOptProblems: getdh
using SparseArrays, Tracker, Statistics, ChainRulesCore, Zygote
using Nonconvex: Nonconvex

export  Objective,
        IneqConstraint,
        BlockIneqConstraint,
        Volume,
        Compliance,
        MeanCompliance,
        BlockCompliance,
        AbstractFunction,
        getfevals,
        getmaxfevals,
        maxedfevals,
        getnvars,
        MicroVonMisesStress,
        MacroVonMisesStress,
        project,
        generate_scenarios,
        hutch_rand!,
        hadamard!

const to = TimerOutput()

abstract type AbstractFunction{T} <: Nonconvex.AbstractFunction end

abstract type AbstractConstraint{T} <: AbstractFunction{T} end

@params struct Objective{T} <: AbstractFunction{T}
    f
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Objective) = println("TopOpt objective function")
Objective(::Type{T}, f) where {T <: Real} = Objective{T, typeof(f)}(f)
Objective(f::AbstractFunction{T}) where {T <: Real} = Objective(T, f)
Objective(f::Function) = Objective(Float64, f)
@forward_property Objective f

Nonconvex.getdim(o::Objective) = Nonconvex.getdim(o.f)

@params struct IneqConstraint{T} <: AbstractConstraint{T}
    f
    s
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::IneqConstraint) = println("TopOpt inequality constraint function")
IneqConstraint(::Type{T}, f, s) where {T} = IneqConstraint{T, typeof(f), typeof(s)}(f, s)
IneqConstraint(f::AbstractFunction{T}, s) where {T} = IneqConstraint(T, f, s)
IneqConstraint(f::Function, s) = IneqConstraint(Float64, f, s)
@forward_property IneqConstraint f
Nonconvex.getdim(c::IneqConstraint) = 1

@params struct BlockIneqConstraint{T} <: AbstractConstraint{T}
    f
    s::Union{T, AbstractVector{T}}
    dim::Int
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::BlockIneqConstraint) = println("TopOpt block inequality constraint")
function BlockIneqConstraint(::Type{T}, f, s, dim = Nonconvex.getdim(f)) where {T}
    return BlockIneqConstraint(f, convert.(T, s), dim)
end
function BlockIneqConstraint(f::AbstractFunction{T}, s::Union{Any, AbstractVector}) where {T}
    return BlockIneqConstraint(f, convert.(T, s), Nonconvex.getdim(f))
end
function BlockIneqConstraint(f::Function, s::Union{Any, AbstractVector})
    return BlockIneqConstraint(f, s, Nonconvex.getdim(f))
end
@forward_property BlockIneqConstraint f
Nonconvex.getdim(c::BlockIneqConstraint) = c.dim
TopOpt.getnvars(c::BlockIneqConstraint) = TopOpt.getnvars(c.f)
(bc::BlockIneqConstraint)(x) = bc.f(x) .- bc.s

Base.broadcastable(o::Union{Objective, AbstractConstraint}) = Ref(o)
getfunction(o::Union{Objective, AbstractConstraint}) = o.f
getfunction(f::AbstractFunction) = f

(o::Objective)(args...) = o.f(args...)

(c::IneqConstraint)(args...) = c.f(args...) - c.s

include("compliance.jl")
include("volume.jl")
include("stress.jl")
include("trace.jl")
include("mean_compliance.jl")
include("block_compliance.jl")

end
