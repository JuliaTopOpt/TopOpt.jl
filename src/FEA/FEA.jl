module FEA

using ..TopOpt: TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..Utilities
using Ferrite, Setfield, TimerOutputs, Preconditioners
using IterativeSolvers, StaticArrays, SparseArrays
using LinearAlgebra
using Parameters: @unpack

export AbstractFEASolver,
    FEASolver,
    DirectSolver,
    CGAssembleSolver,
    CGMatrixFreeSolver,
    DefaultCriteria,
    EnergyCriteria,
    simulate,
    AbstractPhysics,
    LinearElasticity,
    HeatTransfer

const to = TimerOutput()

# FEA solvers
abstract type AbstractFEASolver end

include("solvers_api.jl")  # Shared abstractions first
include("grid_utils.jl")
include("matrix_free_operator.jl")
include("convergence_criteria.jl")
include("matrix_free_apply_bcs.jl")
include("simulate.jl")

getcompliance(solver) = solver.u' * solver.globalinfo.K * solver.u

end
