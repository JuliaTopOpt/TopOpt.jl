module Functions

using ..TopOpt: dim, TopOpt, PENALTY_BEFORE_INTERPOLATION, PseudoDensities
using ..TopOptProblems, ..TrussTopOptProblems
using ..TopOptProblems: initialize_K, getdh
using ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra, Requires
using ..TrussTopOptProblems: getA, compute_local_axes

using Parameters: @unpack
using TimerOutputs, Ferrite, StaticArrays, StatsFuns
using SparseArrays, Statistics, ChainRulesCore, Zygote
using Nonconvex: Nonconvex
using Flux
using AbstractDifferentiation: AbstractDifferentiation
using StatsBase, Statistics 
using Plots:heatmap  
const AD = AbstractDifferentiation

export Volume,
    Compliance,
    Displacement,
    MeanCompliance,
    BlockCompliance,
    AbstractFunction,
    getfevals,
    getmaxfevals,
    maxedfevals,
    von_mises_stress_function,
    project,
    generate_scenarios,
    hutch_rand!,
    hadamard!,
    TrussStress,
    AssembleK,
    TrussElementKσ,
    ElementK,
    apply_boundary_with_zerodiag!,
    apply_boundary_with_meandiag!,
    NeuralNetwork,
    TrainFunction,
    PredictFunction,
    NNParams,
    Coordinates,
    StressTensor,
    ElementStressTensor,
    MaterialInterpolation,
    MultiMaterialVariables,
    element_densities,
    tounit,
    DefGradTensor,
    GoodnessTensor, 
    ElementDefGradTensor, 
    ElementGoodnessTensor, 
    FToK2AndK3,
    Entropy_Calc,
    Entropy,
    SMu_gen, 
    SAlpha_gen 

const to = TimerOutput()

abstract type AbstractFunction{T} <: Nonconvex.NonconvexCore.AbstractFunction end

include("function_utils.jl")
include("compliance.jl")
include("displacement.jl")
include("volume.jl")
include("trace.jl")
include("mean_compliance.jl")
include("block_compliance.jl")

# stress
include("stress_tensor.jl")

# buckling
include("apply_boundary.jl")
include("assemble_K.jl")
include("element_ksigma.jl")
include("element_k.jl")

# Goodness related
include("defgrad.jl")
include("goodness.jl")

# TODO no rrules yet
include("truss_stress.jl")

include("neural.jl")

include("interpolation.jl")

end
