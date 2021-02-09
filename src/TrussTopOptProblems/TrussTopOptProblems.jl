module TrussTopOptProblems

using JuAFEM, StaticArrays, LinearAlgebra
using SparseArrays
using ..TopOpt
using ..TopOpt.Utilities
using Setfield
# using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
# using ..Utilities: @forward_property

abstract type AbstractFEAMaterial end
struct TrussFEAMaterial{T} <: AbstractFEAMaterial
    E::T # Young's modulus
    ν::T # Poisson's ratio
end

abstract type AbstractFEACrossSec end
struct TrussFEACrossSec{T} <: AbstractFEACrossSec
    A::T # cross section area
end

import JuAFEM: assemble!

# TODO apply @params to all the problem struct

# include("utils.jl")
include("grids.jl")
# include("metadata.jl")
include("problem_types.jl")
include("matrices_and_vectors.jl")
include("elementinfo.jl")
include("buckling.jl")
include(joinpath("TrussIO", "TrussIO.jl"))
using .TrussIO
include(joinpath("TrussVisualization", "TrussVisualization.jl"))
using .TrussVisualization

export TrussGrid, TrussProblem, TrussFEACrossSec, TrussFEAMaterial
export parse_truss_json

end # module
