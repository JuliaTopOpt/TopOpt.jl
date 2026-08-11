module Functions

using ..TopOpt: dim, TopOpt, PENALTY_BEFORE_INTERPOLATION, PseudoDensities
using ..TopOptProblems, ..TrussTopOptProblems
using ..TopOptProblems: initialize_K, getdh
using ..FEA, ..CheqFilters
using ..Utilities, ForwardDiff, LinearAlgebra
using ..TrussTopOptProblems: getA, compute_local_axes
using IterativeSolvers: cg!
using Preconditioners: UpdatePreconditioner!

using Parameters: @unpack
using TimerOutputs, Ferrite, StaticArrays, StatsFuns
using SparseArrays, Statistics, ChainRulesCore, Zygote
using Nonconvex: Nonconvex
using DifferentiationInterface
const DI = DifferentiationInterface

export VolumeFun,
    ComplianceFun,
    DisplacementFun,
    MeanComplianceFun,
    BlockComplianceFun,
    AbstractFunction,
    getfevals,
    getmaxfevals,
    maxedfevals,
    von_mises_stress_function,
    project,
    generate_scenarios,
    hutch_rand!,
    hadamard!,
    TrussStressFun,
    AssembleKFun,
    TrussElementKσ,
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

# TODO no rrules yet
include("truss_stress.jl")

include("neural.jl")

include("interpolation.jl")

# Shared element energy kernel (used by Compliance and ThermalCompliance)
include("compute_element_energy.jl")

# Thermal compliance for heat transfer problems
include("thermal_compliance.jl")

# Fixed element projection for black/white handling
include("fixed_element.jl")

# `Volume` is kept unexported to avoid colliding with Makie.Volume; the
# canonical names use a `Fun` suffix.
const Volume = VolumeFun
const Compliance = ComplianceFun
const Displacement = DisplacementFun
const MeanCompliance = MeanComplianceFun
const BlockCompliance = BlockComplianceFun
const ThermalCompliance = ThermalComplianceFun
const TrussStress = TrussStressFun
const AssembleK = AssembleKFun
const TrussElementKσ = TrussElementKσFun
const ElementK = ElementKFun
const StressTensor = StressTensorFun
const ElementStressTensor = ElementStressTensorFun
const NeuralNetwork = NeuralNetworkFun
const TrainFunction = TrainFunctionFun
const PredictFunction = PredictFunctionFun
const MaterialInterpolation = MaterialInterpolationFun
const MultiMaterialVariables = MultiMaterialVariablesFun
const FixedElementProjector = FixedElementProjectorFun
export Compliance,
    Displacement,
    MeanCompliance,
    BlockCompliance,
    ThermalCompliance,
    TrussStress,
    AssembleK,
    TrussElementKσ,
    ElementK,
    StressTensor,
    ElementStressTensor,
    NeuralNetwork,
    TrainFunction,
    PredictFunction,
    MaterialInterpolation,
    MultiMaterialVariables,
    FixedElementProjector,
    VolumeFun,
    ComplianceFun,
    DisplacementFun,
    MeanComplianceFun,
    BlockComplianceFun,
    ThermalComplianceFun,
    TrussStressFun,
    AssembleKFun,
    TrussElementKσFun,
    ElementKFun,
    StressTensorFun,
    ElementStressTensorFun,
    NeuralNetworkFun,
    TrainFunctionFun,
    PredictFunctionFun,
    MaterialInterpolationFun,
    MultiMaterialVariablesFun,
    FixedElementProjectorFun

end
