module Functions

using ..TopOpt: TopOpt, PENALTY_BEFORE_INTERPOLATION, PseudoDensities
using ..TopOptProblems, ..TrussTopOptProblems
using ..TopOptProblems: initialize_K, getdh
using ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra
using ..Utilities: get_ρ, get_ρ_dρ
using ..TrussTopOptProblems: getA, compute_local_axes
using IterativeSolvers: cg!
using Preconditioners: UpdatePreconditioner!

using Parameters: @unpack
using Ferrite, StaticArrays, SparseArrays, Statistics, ChainRulesCore
using Nonconvex: Nonconvex
using DifferentiationInterface
const DI = DifferentiationInterface

export VolumeFun,
    ComplianceFun,
    DisplacementFun,
    MeanComplianceFun,
    BlockComplianceFun,
    AbstractFunction,
    von_mises_stress_function,
    generate_scenarios,
    hutch_rand!,
    hadamard!,
    TrussStressFun,
    AssembleKFun,
    TrussElementKσFun,
    ElementKFun,
    apply_boundary_with_zerodiag!,
    apply_boundary_with_meandiag!,
    NeuralNetworkFun,
    TrainFunctionFun,
    PredictFunctionFun,
    NNParams,
    Coordinates,
    AbstractMLModel,
    getcentroids,
    StressTensorFun,
    ElementStressTensorFun,
    MaterialInterpolationFun,
    MultiMaterialVariablesFun,
    element_densities,
    tounit,
    ThermalComplianceFun,
    FixedElementProjectorFun,
    get_fixed_element_projector,
    get_free_variables,
    get_free_variable_count

abstract type AbstractFunction{T} <: Nonconvex.NonconvexCore.AbstractFunction end

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

# TODO no rrules yet
include("truss_stress.jl")

include("neural.jl")

include("interpolation.jl")

# Shared element energy kernel (used by ComplianceFun and ThermalComplianceFun)
include("compute_element_energy.jl")

# Thermal compliance for heat transfer problems
include("thermal_compliance.jl")

# Fixed element projection for black/white handling
include("fixed_element.jl")

end
