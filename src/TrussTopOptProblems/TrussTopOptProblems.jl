module TrussTopOptProblems

using Ferrite, StaticArrays, LinearAlgebra
using SparseArrays
using ..TopOpt
using ..TopOpt.Utilities
using ..TopOpt.TopOptProblems: _base_interpolation
using Setfield
import Ferrite: assemble!
using LinearAlgebra: I, norm
using NearestNeighbors

abstract type AbstractFEAMaterial end
"""
    TrussFEAMaterial(E, ν)

Material container for truss FEA: Young's modulus `E` and Poisson's ratio `ν`.
"""
struct TrussFEAMaterial{T} <: AbstractFEAMaterial
    E::T # Young's modulus
    ν::T # Poisson's ratio
end

abstract type AbstractFEACrossSec end
"""
    TrussFEACrossSec(A)

Cross-section container for truss FEA with area `A`.
"""
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
