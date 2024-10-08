module TopOpt

const PENALTY_BEFORE_INTERPOLATION = true
using Reexport, ChainRulesCore

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
    return PseudoDensities{I,P,F,T,N,A}(x), Δ -> (NoTangent(), Δ isa Tangent ? Δ.x : Δ)
end

Base.BroadcastStyle(::Type{T}) where {T<:PseudoDensities} = Broadcast.ArrayStyle{T}()
function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}, ::Type{ElType}
) where {T,ElType}
    return similar(T, axes(bc))
end
function Base.similar(
    ::Type{<:TV}, axes::Tuple{Union{Integer,Base.OneTo},Vararg{Union{Integer,Base.OneTo}}}
) where {I,P,F,T,N,A,TV<:PseudoDensities{I,P,F,T,N,A}}
    return PseudoDensities{I,P,F}(similar(A, axes))
end

Base.length(x::PseudoDensities) = length(x.x)
Base.size(x::PseudoDensities, i...) = size(x.x, i...)
Base.getindex(x::PseudoDensities, i...) = x.x[i...]
Base.sum(x::PseudoDensities) = sum(x.x)
LinearAlgebra.dot(x::PseudoDensities, weights::AbstractArray) = dot(x.x, weights)

export PseudoDensities

# Utilities
include(joinpath("Utilities", "Utilities.jl"))
using .Utilities

function visualize(arg::T; kwargs...) where {T}
    return error(
        "`visualize` is not defined for input type `$T`. This may be because the input to the function is incorrect or because you forgot to load `Makie` in your code. You can load `Makie` with `using Makie`. To see the available methods of `visualize` and their documentation, you can run `? visualize` in the Julia REPL.",
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

using ForwardDiff, IterativeSolvers#, Preconditioners
@reexport using VTKDataTypes

const DEBUG = Base.RefValue(false)

# FEA solvers
include(joinpath("FEA", "FEA.jl"))
using .FEA

# Chequeurboard filter
include(joinpath("CheqFilters", "CheqFilters.jl"))
using .CheqFilters

# Objective and constraint functions
@reexport using Flux
include(joinpath("Functions", "Functions.jl"))
@reexport using .Functions

# Various topology optimization algorithms
include(joinpath("Algorithms", "Algorithms.jl"))
using .Algorithms

export TopOpt,
    simulate,
    TopOptTrace,
    DirectDisplacementSolver,
    PCGDisplacementSolver,
    StaticMatrixFreeDisplacementSolver,
    SensFilter,
    DensityFilter,
    Displacement,
    Compliance,
    CG,
    Direct,
    Assembly,
    MatrixFree,
    FEASolver,
    Optimizer,
    SIMP,
    ContinuationSIMP,
    BESO,
    GESO,
    PowerContinuation,
    ExponentialContinuation,
    LogarithmicContinuation,
    CubicSplineContinuation,
    SigmoidContinuation,
    Continuation,
    save_mesh,
    DefaultCriteria,
    EnergyCriteria,
    PowerPenalty,
    RationalPenalty,
    SinhPenalty,
    MMA87,
    MMA02,
    HeavisideProjection,
    SigmoidProjection,
    ProjectedPenalty,
    visualize
end
