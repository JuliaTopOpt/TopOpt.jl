module TopOptProblems

using Ferrite, StaticArrays, LinearAlgebra
using SparseArrays, Setfield, Requires
using ..TopOpt.Utilities
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: @forward_property
using Distributions: Distributions

using VTKDataTypes

import Ferrite: assemble!
const QuadraticHexahedron = Ferrite.Cell{3,20,6}

abstract type AbstractTopOptProblem end

include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("multiload.jl")
include("elementmatrix.jl")
include("matrices_and_vectors.jl")
include("elementinfo.jl")
include("assemble.jl")
include("buckling.jl")

include(joinpath("IO", "IO.jl"))
using .InputOutput

include("Visualization/Visualization.jl")
using .Visualization

export RayProblem,
    PointLoadCantilever,
    HalfMBB,
    LBeam,
    TieBeam,
    InpStiffness,
    StiffnessTopOptProblem,
    AbstractTopOptProblem,
    GlobalFEAInfo,
    ElementFEAInfo,
    YoungsModulus,
    assemble,
    assemble_f!,
    RaggedArray,
    ElementMatrix,
    rawmatrix,
    bcmatrix,
    save_mesh,
    RandomMagnitude,
    MultiLoad,
    TensionBar

end # module
