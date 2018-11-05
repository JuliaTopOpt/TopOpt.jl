module TopOpt

using Reexport, Parameters
using TopOptProblems
using JuAFEM, StaticArrays
using MMA, ForwardDiff
using Optim
using IterativeSolvers#, Preconditioners
using IntervalArithmetic, IntervalRootFinding
using IntervalOptimisation

using TimerOutputs

const to = TimerOutput()
const DEBUG = Base.RefValue(false)

#norm(a) = sqrt(dot(a,a))

# Utilities
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
#include(joinpath("fea_solvers", "assemble.jl"))
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

# Topology optimization algorithms
abstract type TopOptAlgorithm end
abstract type AbstractSIMP <: TopOptAlgorithm end

## Traditional SIMP
include(joinpath("simp", "simp.jl"))

## Continuation SIMP
include(joinpath("simp", "continuation_schemes.jl"))
include(joinpath("simp", "continuation_simp.jl"))

## Adaptive SIMP
include(joinpath("simp", "polynomials.jl"))
include(joinpath("simp", "polynomials2.jl"))
include(joinpath("simp", "adaptive_simp.jl"))

## BESO
include(joinpath(".", "beso.jl"))

## GESO
include(joinpath(".","geso.jl"))

# Export
include(joinpath(".", "writevtk.jl"))

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
