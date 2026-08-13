module FEA

using ..TopOpt: TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..Utilities
using Ferrite, Setfield, Preconditioners
using IterativeSolvers, StaticArrays, SparseArrays
using LinearAlgebra
using Parameters: @unpack

export AbstractFEASolver,
    FEASolver,
    DirectSolver,
    CGAssemblySolver,
    CGMatrixFreeSolver,
    DefaultCriteria,
    EnergyCriteria,
    ConvergenceCriteria,
    simulate,
    AbstractPhysics,
    LinearElasticity,
    HeatTransfer,
    MatrixFreeOperator,
    MatrixOperator,
    SolverResult

# FEA solvers
"""
    AbstractFEASolver

Abstract type for all FEA solvers in TopOpt. `GenericFEASolver` is the concrete
implementation.
"""
abstract type AbstractFEASolver end

include("solvers_api.jl")  # Shared abstractions first
include("matrix_free_operator.jl")
include("convergence_criteria.jl")
include("matrix_free_apply_bcs.jl")

getcompliance(solver) = solver.u' * solver.globalinfo.K * solver.u

end
