using JuAFEM
using TopOpt
using IterativeSolvers
using Preconditioners
using StaticArrays

abstract type AbstractFEASolver end

include("traces.jl")
include("penalties.jl")

include("direct_displacement_solver.jl")
include("assembly_cg_displacement_solvers.jl")
include("matrix_free_operator.jl")
include("matrix_free_cg_displacement_solvers.jl")

include("cheqfilters2.jl")
include("objectives.jl")

include("constraints.jl")
