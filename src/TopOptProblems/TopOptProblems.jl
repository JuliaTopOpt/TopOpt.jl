module TopOptProblems

using JuAFEM, StaticArrays, LinearAlgebra
using SparseArrays, Setfield, CuArrays
using CUDAnative
using ..GPUUtils, ..Utilities
using CUDAdrv: CUDAdrv
#using Makie
#using GeometryTypes

import JuAFEM: assemble!
import ..GPUUtils: whichdevice

abstract type AbstractTopOptProblem end

const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

include("utils.jl")
include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("matrices_and_vectors.jl")
include("assemble.jl")
include(joinpath("IO", "IO.jl"))
using .IO
#include("makie.jl")

@define_cu(ElementFEAInfo, :Kes, :fes, :fixedload, :cellvolumes, :metadata, :black, :white, :varind, :cells)
@define_cu(TopOptProblems.Metadata, :cell_dofs, :dof_cells, :node_cells, :node_dofs)
@define_cu(JuAFEM.ConstraintHandler, :values, :prescribed_dofs, :dh)
@define_cu(JuAFEM.DofHandler, :grid)
@define_cu(JuAFEM.Grid, :cells)
for T in (PointLoadCantilever, HalfMBB, LBeam, TieBeam, InpStiffness)
    @eval @define_cu($T, :ch, :black, :white, :varind)
end

export PointLoadCantilever, HalfMBB, LBeam, TieBeam, InpStiffness, StiffnessTopOptProblem, AbstractTopOptProblem, GlobalFEAInfo, ElementFEAInfo, YoungsModulus, assemble, assemble_f!, RaggedArray, ElementMatrix, rawmatrix, bcmatrix, save_mesh

end # module
