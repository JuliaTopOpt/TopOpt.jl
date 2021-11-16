module Functions

using ..TopOpt: dim, whichdevice, CPU, GPU, TopOpt, PENALTY_BEFORE_INTERPOLATION
using ..TopOptProblems, ..TrussTopOptProblems
using ..TopOptProblems: initialize_K, getdh
using ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra, Requires
using ..TrussTopOptProblems: getA, compute_local_axes

using Parameters: @unpack
using TimerOutputs, Ferrite, StaticArrays
using StatsFuns, MappedArrays, LazyArrays
using SparseArrays, Statistics, ChainRulesCore, Zygote
using Nonconvex: Nonconvex

export  Volume,
        Compliance,
        Displacement,
        MeanCompliance,
        BlockCompliance,
        AbstractFunction,
        getfevals,
        getmaxfevals,
        maxedfevals,
        MicroVonMisesStress,
        MacroVonMisesStress,
        project,
        generate_scenarios,
        hutch_rand!,
        hadamard!,
        TrussStress,
        AssembleK,
        TrussElementKσ,
        ElementK,
        apply_boundary_with_zerodiag!,
        apply_boundary_with_meandiag!

const to = TimerOutput()

abstract type AbstractFunction{T} <: Nonconvex.NonconvexCore.AbstractFunction end

include("function_utils.jl")
include("compliance.jl")
include("displacement.jl")
include("volume.jl")
include("stress.jl")
include("trace.jl")
include("mean_compliance.jl")
include("block_compliance.jl")

# buckling
include("apply_boundary.jl")
include("assemble_K.jl")
include("element_ksigma.jl")
include("element_k.jl")

# TODO no rrules yet
include("truss_stress.jl")

end
