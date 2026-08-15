module TopOpt

using Reexport, ChainRulesCore, Preferences

"""
    PENALTY_BEFORE_INTERPOLATION

Package preference for penalty application order.
Default is `true` (penalty applied before density interpolation).
Can be set using `Preferences.jl` with:
    using Preferences, TopOpt
    set_preferences!(TopOpt, "penalty_before_interpolation" => true/false)

Note: Changing this preference requires restarting Julia and recompiling the package.
"""
const PENALTY_BEFORE_INTERPOLATION = let
    pref = @load_preference("penalty_before_interpolation", "true")
    if pref isa Bool
        pref
    else
        parse(Bool, pref)
    end
end

@reexport using Nonconvex, NonconvexMMA, NonconvexSemidefinite, NonconvexPercival

# I: interpolated
# P: penalized
# F: filtered
struct PseudoDensities{I,P,F,T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    x::A
end
function Base.setindex!(A::PseudoDensities, x, inds...)
    return A.x[inds...] = x
end
function PseudoDensities(x::A) where {T,N,A<:AbstractArray{T,N}}
    return PseudoDensities{false,false,false,T,N,A}(x)
end
function PseudoDensities{I,P,F}(x::A) where {I,P,F,T,N,A<:AbstractArray{T,N}}
    return PseudoDensities{I,P,F,T,N,A}(x)
end
function ChainRulesCore.rrule(
    ::Type{PseudoDensities{I,P,F,T,N,A}}, x
) where {I,P,F,T,N,A<:AbstractArray{T,N}}
    return PseudoDensities{I,P,F,T,N,A}(x), Δ -> begin
        Δ = ChainRulesCore.unthunk(Δ)
        (NoTangent(), Δ isa Tangent ? Δ.x : Δ)
    end
end

Base.BroadcastStyle(::Type{T}) where {T<:PseudoDensities} = Broadcast.ArrayStyle{T}()
function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}, ::Type{ElType}
) where {T<:PseudoDensities,ElType}
    return similar(T, axes(bc))
end
function Base.similar(
    ::Type{<:TV}, axes::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}}
) where {I,P,F,T,N,A,TV<:PseudoDensities{I,P,F,T,N,A}}
    return PseudoDensities{I,P,F}(similar(A, axes))
end

Base.length(x::PseudoDensities) = length(x.x)
Base.size(x::PseudoDensities, i...) = size(x.x, i...)
Base.getindex(x::PseudoDensities, i::Integer...) = x.x[i...]
Base.getindex(x::PseudoDensities, i::CartesianIndex) = x.x[i]
Base.sum(x::PseudoDensities) = sum(x.x)

# Resolve ambiguity with Base.similar(::Type{<:AbstractArray}, ::NTuple{N,Int})
function Base.similar(
    ::Type{TV}, dims::Tuple{Int,Vararg{Int}}
) where {I,P,F,T,N,A,TV<:PseudoDensities{I,P,F,T,N,A}}
    return PseudoDensities{I,P,F}(similar(A, dims))
end

export PseudoDensities

# Utilities
include(joinpath("Utilities", "Utilities.jl"))
using .Utilities

function visualize(arg::T; kwargs...) where {T}
    # Heuristic: if a Makie backend is loaded but the extension still hasn't
    # precompiled (e.g. the user's precompile cache predates their current
    # Makie version), surface `retry_load_extensions!` so the fix is one line.
    if isdefined(Main, :Makie) ||
        isdefined(Main, :WGLMakie) ||
        isdefined(Main, :GLMakie) ||
        isdefined(Main, :CairoMakie)
        Base.retry_load_extensions()
    end
    return error(
        "`visualize` is not defined for input type `$T`. The Makie-backed " *
        "methods of `visualize` live in the `TopOptMakieExt` extension, " *
        "which loads automatically when a Makie backend (or `using Makie`) is " *
        "in your active environment alongside `using TopOpt`. Common fixes:" *
        "\n  · `using TopOpt` then `using Makie` (or `using WGLMakie`) in the " *
        "same session\n  · `using TopOpt, WGLMakie` to load both at once\n" *
        "  · Run `Base.retry_load_extensions()` if both are loaded but the " *
        "extension still isn't active.",
    )
end

function visualize_static(arg::T; kwargs...) where {T}
    if isdefined(Main, :Makie) ||
        isdefined(Main, :WGLMakie) ||
        isdefined(Main, :GLMakie) ||
        isdefined(Main, :CairoMakie)
        Base.retry_load_extensions()
    end
    return error(
        "`visualize_static` is not defined for input type `$T`. " *
        "It requires the WGLMakie backend: load it with `using WGLMakie` " *
        "(after `using TopOpt`). For static (non-live) rendering pass " *
        "`exportable=true, offline=true` to `Bonito.Page` first.",
    )
end

# Topopology optimization problem definitions
include(joinpath("TopOptProblems", "TopOptProblems.jl"))

using LinearAlgebra, Statistics
using Reexport, Parameters, Setfield
@reexport using .TopOptProblems

# Truss Topopology optimization problem definitions
include(joinpath("TrussTopOptProblems", "TrussTopOptProblems.jl"))
@reexport using .TrussTopOptProblems

using Ferrite, StaticArrays
using Ferrite: getncells
export getncells

using ForwardDiff, IterativeSolvers#, Preconditioners
using VTKDataTypes
using VTKDataTypes: VTKUnstructuredData

const DEBUG = Base.RefValue(false)

# FEA solvers
include(joinpath("FEA", "FEA.jl"))
using .FEA

# Chequeurboard filter
include(joinpath("CheqFilters", "CheqFilters.jl"))
using .CheqFilters

# Objective and constraint functions
include(joinpath("Functions", "Functions.jl"))
@reexport using .Functions

# Various topology optimization algorithms
include(joinpath("Algorithms", "Algorithms.jl"))
using .Algorithms

export TopOpt,
    simulate,
    SensFilterFun,
    DensityFilterFun,
    DisplacementFun,
    ComplianceFun,
    ThermalComplianceFun,
    FEASolver,
    DirectSolver,
    CGAssemblySolver,
    CGMatrixFreeSolver,
    MatrixFreeOperator,
    MatrixOperator,
    BESO,
    GESO,
    save_mesh,
    DefaultCriteria,
    EnergyCriteria,
    PowerPenaltyFun,
    RationalPenaltyFun,
    SinhPenaltyFun,
    MMA87,
    MMA02,
    HeavisideProjectionFun,
    SigmoidProjectionFun,
    ProjectedPenaltyFun,
    setpenalty!,
    visualize,
    visualize_static
end
