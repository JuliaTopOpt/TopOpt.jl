module TrussTopOptProblems

using Ferrite, StaticArrays, LinearAlgebra
using SparseArrays
using ..TopOpt
using ..TopOpt.Utilities
using ..TopOpt.Functions: apply_boundary_with_zerodiag!, AssembleK
using Setfield
import Ferrite: assemble!
using LinearAlgebra: I, norm

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
include("buckling.jl")
include(joinpath("TrussIO", "TrussIO.jl"))
using .TrussIO
include(joinpath("TrussVisualization", "TrussVisualization.jl"))
using .TrussVisualization

export TrussGrid, TrussProblem, TrussFEACrossSec, TrussFEAMaterial
export load_truss_geo, load_truss_json

end # module
