module TopOpt

const PENALTY_BEFORE_INTERPOLATION = true
using Requires, Reexport

macro cuda_only(mod, code)
    quote
        @init @require CUDASupport = "97986420-7ec3-11e9-24cd-4f0e301eb539" @eval $mod begin
            $code
        end
    end |> esc
end

abstract type AbstractDevice end
struct CPU <: AbstractDevice end
struct GPU <: AbstractDevice end
whichdevice(::Any) = CPU()

# GPU utilities
module GPUUtils end

# Utilities
include(joinpath("Utilities", "Utilities.jl"))
using .Utilities

# Topopology optimization problem definitions
include(joinpath("TopOptProblems", "TopOptProblems.jl"))

using LinearAlgebra, Statistics
using Reexport, Parameters, Setfield
@reexport using .TopOptProblems, Optim, LineSearches
using JuAFEM, StaticArrays

using ForwardDiff, IterativeSolvers#, Preconditioners
@reexport using VTKDataTypes

const DEBUG = Base.RefValue(false)
function dim end
getnvars(f) = length(f.grad)
function jtvp! end
#norm(a) = sqrt(dot(a,a))

# FEA solvers
include(joinpath("FEA", "FEA.jl"))
using .FEA

# Chequeurboard filter
include(joinpath("CheqFilters", "CheqFilters.jl"))
using .CheqFilters

# Objective and constraint functions
include(joinpath("Functions", "Functions.jl"))
@reexport using .Functions

# Method of moving asymptotes
include(joinpath("MMA", "MMA.jl"))
@reexport using .MMA

# Flux optimisers
include(joinpath("FluxOptimisers", "Optimise.jl"))

# Hybrid method of moving asymptotes and augmented Lagrangian
include(joinpath("MMALag", "MMALag.jl"))
@reexport using .MMALag

# Various topology optimization algorithms
include(joinpath("Algorithms", "Algorithms.jl"))
using .Algorithms

include("AugLag/AugLag.jl")
using .AugLag

macro init_cuda()
    quote
        const CuArrays = CUDASupport.CuArrays
        const CUDAdrv = CUDASupport.CUDAdrv
        const CUDAnative = CUDASupport.CUDAnative
        const GPUArrays = CUDASupport.GPUArrays
        CuArrays.allowscalar(false)
        const dev = CUDAdrv.device()
        const ctx = CUDAdrv.CuContext(dev)
        using .CuArrays, .CUDAnative
        using .GPUArrays: GPUVector, GPUArray
    end |> esc
end

@cuda_only GPUUtils include("GPUUtils/GPUUtils.jl")
@cuda_only Utilities include("Utilities/gpu_utilities.jl")
@cuda_only MMA include("MMA/gpu_mma.jl")
@cuda_only TopOptProblems include("TopOptProblems/gpu_support.jl")
@cuda_only FEA include("FEA/gpu_solvers.jl")
@cuda_only CheqFilters include("CheqFilters/gpu_cheqfilter.jl")
@cuda_only Functions include("Functions/gpu_support.jl")
@cuda_only Algorithms include("Algorithms/SIMP/gpu_simp.jl")

export  TopOpt,
        simulate, 
        TopOptTrace, 
        Objective,
        Constraint,
        Zero,
        Volume,
        DirectDisplacementSolver,
        PCGDisplacementSolver,
        StaticMatrixFreeDisplacementSolver,
        SensFilter,
        DensityFilter,
        Compliance,
        MeanCompliance,
        BlockCompliance,
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
        Continuation,
        save_mesh,
        CPU,
        GPU,
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
