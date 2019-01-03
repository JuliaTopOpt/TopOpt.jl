module TopOpt

# GPU utilities
include(joinpath("GPUUtils", "GPUUtils.jl"))

# Method of moving asymptotes
include(joinpath("MMA", "MMA.jl"))

# Topopology optimization problem definitions
include(joinpath("TopOptProblems", "TopOptProblems.jl"))

using LinearAlgebra, Statistics
using Reexport, Parameters, Setfield, .GPUUtils
@reexport using .TopOptProblems, Optim, .MMA, LineSearches
using JuAFEM, StaticArrays, CuArrays, CUDAnative, GPUArrays
using CUDAdrv: CUDAdrv
using ForwardDiff, IterativeSolvers#, Preconditioners

CuArrays.allowscalar(false)
const DEBUG = Base.RefValue(false)
const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

#norm(a) = sqrt(dot(a,a))

# Utilities
include(joinpath("Utilities", "Utilities.jl"))
using .Utilities

# FEA solvers
include(joinpath("FEA", "FEA.jl"))
using .FEA

# Chequeurboard filter
include(joinpath("CheqFilters", "CheqFilters.jl"))
using .CheqFilters

# Objective and constraint functions
include(joinpath("Functions", "Functions.jl"))
using .Functions

# Various topology optimization algorithms
include(joinpath("Algorithms", "Algorithms.jl"))
using .Algorithms

# Inp file parser
include(joinpath("InpParser", "InpParser.jl"))

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
