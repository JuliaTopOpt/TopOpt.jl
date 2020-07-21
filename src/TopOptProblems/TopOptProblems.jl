module TopOptProblems

using JuAFEM, StaticArrays, LinearAlgebra
using SparseArrays, Setfield, Requires
using ..TopOpt.Utilities
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: @forward_property
import Distributions

using VTKDataTypes
#using Makie
#using GeometryTypes

import JuAFEM: assemble!

abstract type AbstractTopOptProblem end

include("utils.jl")
include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("multiload.jl")
include("matrices_and_vectors.jl")
include("assemble.jl")
include(joinpath("IO", "IO.jl"))
using .IO
include("makie.jl")

export PointLoadCantilever, HalfMBB, LBeam, TieBeam, InpStiffness, StiffnessTopOptProblem, AbstractTopOptProblem, GlobalFEAInfo, ElementFEAInfo, YoungsModulus, assemble, assemble_f!, RaggedArray, ElementMatrix, rawmatrix, bcmatrix, save_mesh, RandomMagnitude, MultiLoad

end # module
