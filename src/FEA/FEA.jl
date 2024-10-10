module FEA

using ..TopOpt: TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..Utilities, Requires
using Ferrite, Setfield, TimerOutputs, Preconditioners
using IterativeSolvers, StaticArrays, SparseArrays
using LinearAlgebra
using Parameters: @unpack

export AbstractFEASolver,
    AbstractDisplacementSolver,
    AbstractHyperelasticSolver,
    DirectDisplacementSolver,
    PCGDisplacementSolver,
    StaticMatrixFreeDisplacementSolver,
    HyperelasticDisplacementSolver,
    Displacement,
    Direct,
    CG,
    MatrixFree,
    Hyperelastic,
    FEASolver,
    Assembly,
    DefaultCriteria,
    EnergyCriteria,
    simulate

const to = TimerOutput()

# FEA solvers
abstract type AbstractFEASolver end

include("grid_utils.jl")
include("matrix_free_operator.jl")
include("convergence_criteria.jl")
include("direct_displacement_solver.jl")
include("assembly_cg_displacement_solvers.jl")
include("matrix_free_cg_displacement_solvers.jl")
include("matrix_free_apply_bcs.jl")
include("iterative_hyperelastic_solver.jl")
include("simulate.jl")
include("solvers_api.jl")

getcompliance(solver) = solver.u' * solver.globalinfo.K * solver.u

end
