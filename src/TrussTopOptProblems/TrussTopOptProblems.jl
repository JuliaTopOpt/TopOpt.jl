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

# Treat `TrussFEACrossSec` as a scalar-like wrapper over its area so it can be
# used as a design variable (e.g. `TrussFEACrossSec(1.0) + 1`).
Base.:+(a::TrussFEACrossSec, b::Number) = TrussFEACrossSec(a.A + b)
Base.:+(a::Number, b::TrussFEACrossSec) = TrussFEACrossSec(a + b.A)
Base.:+(a::TrussFEACrossSec, b::TrussFEACrossSec) = TrussFEACrossSec(a.A + b.A)
Base.:-(a::TrussFEACrossSec, b::Number) = TrussFEACrossSec(a.A - b)
Base.:-(a::TrussFEACrossSec) = TrussFEACrossSec(-a.A)
Base.:-(a::TrussFEACrossSec, b::TrussFEACrossSec) = TrussFEACrossSec(a.A - b.A)
Base.:*(a::TrussFEACrossSec, b::Number) = TrussFEACrossSec(a.A * b)
Base.:*(a::Number, b::TrussFEACrossSec) = TrussFEACrossSec(a * b.A)
Base.:/(a::TrussFEACrossSec, b::Number) = TrussFEACrossSec(a.A / b)
Base.zero(::TrussFEACrossSec{T}) where {T} = TrussFEACrossSec(zero(T))
Base.one(::TrussFEACrossSec{T}) where {T} = TrussFEACrossSec(one(T))
Base.eltype(::Type{<:TrussFEACrossSec{T}}) where {T} = T
Base.eltype(::TrussFEACrossSec{T}) where {T} = T

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
