module TopOptProblems

using Ferrite, StaticArrays, LinearAlgebra
using SparseArrays, Setfield
using ..TopOpt.Utilities
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: @forward_property
using Distributions: Distributions, Uniform

using VTKDataTypes

import Ferrite: assemble!

"""
    AbstractTopOptProblem

Abstract supertype for all topology optimization problems (continuum
`StiffnessTopOptProblem`, `HeatTransferTopOptProblem`, and truss
`TrussProblem`). Every concrete subtype provides a `Ferrite.ConstraintHandler`
(`ch`), element metadata, and the accessors in this module (`getdim`, `getdh`,
`getncells`, ...).
"""
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

export PointLoadCantilever,
    HalfMBB,
    LBeam,
    TieBeam,
    InpStiffness,
    StiffnessTopOptProblem,
    HeatTransferTopOptProblem,
    HeatConductionProblem,
    HeatTree,
    AbstractTopOptProblem,
    getk,
    getpressuredict,
    getfacesets,
    getcloaddict,
    getdh,
    GlobalFEAInfo,
    ElementFEAInfo,
    YoungsModulus,
    PoissonRatio,
    assemble,
    assemble_f!,
    buckling,
    get_Kσs,
    RaggedArray,
    ElementMatrix,
    rawmatrix,
    bcmatrix,
    save_mesh,
    RandomMagnitudeFun,
    MultiLoad

end # module
