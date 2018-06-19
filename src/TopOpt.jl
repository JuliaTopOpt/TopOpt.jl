module TopOpt

using Reexport
@reexport using TopOptProblems
using JuAFEM
using TimerOutputs
using ForwardDiff
using MMA
@reexport using Optim
using IterativeSolvers
using Preconditioners
using StaticArrays
using Parameters

#norm(a) = sqrt(dot(a,a))

include("utils.jl")

# Trace definition
include("traces.jl")

# Penalty definitions
include("penalties.jl")

# FEA solvers
abstract type AbstractFEASolver end
include(joinpath("fea_solvers", "direct_displacement_solver.jl"))
include(joinpath("fea_solvers", "assembly_cg_displacement_solvers.jl"))
include(joinpath("fea_solvers", "matrix_free_operator.jl"))
include(joinpath("fea_solvers", "matrix_free_cg_displacement_solvers.jl"))
include(joinpath("fea_solvers", "assemble.jl"))
include(joinpath("fea_solvers", "matrix_free_apply_bcs.jl"))
include(joinpath("fea_solvers", "simulate.jl"))
include(joinpath("fea_solvers", "solvers_api.jl"))

# Chequeurboard filter
include("cheqfilters.jl")

# Objectives
include("objectives.jl")

# Constraints
include("constraints.jl")

# Mathematical programming solver wrappers
include(joinpath("optimizers", "math_optimizers.jl"))

# Continuation SIMP
include(joinpath("simp", "simp.jl"))
include(joinpath("simp", "continuation_schemes.jl"))
include(joinpath("simp", "continuation_simp.jl"))

# BESO
include("beso.jl")

# GESO
include("geso.jl")

# Export
include("writevtk.jl")

export  simulate, 
        TopOptTrace, 
        VolConstr, 
        DirectDisplacementSolver, 
        PCGDisplacementSolver, 
        StaticMatrixFreeDisplacementSolver, 
        CheqFilter, 
        ComplianceObj, 
        Displacement, 
        CG, 
        Direct, 
        Assembly, 
        MatrixFree, 
        FEASolver, 
        MMAOptimizer, 
        SIMP, 
        ContinuationSIMP, 
        BESO,
        GESO,
        PowerContinuation, 
        ExponentialContinuation, 
        LogarithmicContinuation, 
        CubicSplineContinuation, 
        SigmoidContinuation, 
        save_mesh

end