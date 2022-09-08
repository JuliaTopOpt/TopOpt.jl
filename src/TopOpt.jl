module TopOpt

const PENALTY_BEFORE_INTERPOLATION = true
using Requires, Reexport

@reexport using Nonconvex, NonconvexMMA, NonconvexSemidefinite, NonconvexPercival

struct PseudoDensities{I,P,F,T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    x::A
    function PseudoDensities(x::A) where {T,N,A<:AbstractArray{T,N}}
        return new{false,false,false,T,N,A}(x)
    end
    function PseudoDensities{I,P,F}(x::A) where {I,P,F,T,N,A<:AbstractArray{T,N}}
        return new{I,P,F,T,N,A}(x)
    end
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
    ProjectedPenalty,
    PowerPenalty
end
