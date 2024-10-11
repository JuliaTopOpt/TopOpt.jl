module TrussTopOptProblems

using Ferrite, StaticArrays, LinearAlgebra
using SparseArrays
using ..TopOpt
using ..TopOpt.Utilities
using Setfield
import Ferrite: assemble!
using LinearAlgebra: I, norm
using NearestNeighbors

abstract type AbstractFEAMaterial end
struct TrussFEAMaterial{T} <: AbstractFEAMaterial
    E::T # Young's modulus
    ν::T # Poisson's ratio
end

abstract type AbstractFEACrossSec end
struct TrussFEACrossSec{T} <: AbstractFEACrossSec
    A::T # cross section area
end

include("grids.jl")
include("problem_types.jl")
include("matrices_and_vectors.jl")
include("elementinfo.jl")
include(joinpath("TrussIO", "TrussIO.jl"))
using .TrussIO

export TrussGrid, TrussProblem, TrussFEACrossSec, TrussFEAMaterial
export PointLoadCantileverTruss
export load_truss_geo, load_truss_json

end # module
