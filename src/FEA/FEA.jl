module FEA

using ..GPUUtils, ..TopOptProblems, ..Utilities
using JuAFEM, Setfield, TimerOutputs, Preconditioners
using IterativeSolvers, CuArrays, StaticArrays
using LinearAlgebra
import CUDAdrv
using Parameters: @unpack

const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

export  AbstractFEASolver,
        AbstractDisplacementSolver,
        DirectDisplacementSolver,
        PCGDisplacementSolver,
        StaticMatrixFreeDisplacementSolver,
        Displacement,
        Direct,
        CG,
        MatrixFree,
        FEASolver,
        Assembly

const to = TimerOutput()

# FEA solvers
abstract type AbstractFEASolver end

include("direct_displacement_solver.jl")
include("assembly_cg_displacement_solvers.jl")
include("matrix_free_operator.jl")
include("matrix_free_cg_displacement_solvers.jl")
include("matrix_free_apply_bcs.jl")
include("simulate.jl")
include("solvers_api.jl")

for T in (:DirectDisplacementSolver, :PCGDisplacementSolver)
    @eval @inline CuArrays.cu(p::$T) = error("$T does not support the GPU.")
end

end

