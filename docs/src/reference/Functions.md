# `Functions`

This sub-module of `TopOpt` defines the differentiable building blocks used in
topology optimization formulations. The narrative descriptions and constructor
examples live in the [Functions](@ref) page; the entries below render the full
docstrings.

```@meta
CurrentModule = TopOpt.Functions
```

## Abstract type

```@docs
AbstractFunction
```

## ComplianceFun and volume

```@docs
ComplianceFun
VolumeFun
ThermalComplianceFun
MeanComplianceFun
BlockComplianceFun
```

## DisplacementFun and stress

```@docs
DisplacementFun
StressTensorFun
ElementStressTensorFun
von_mises_stress_function
epsilon_relaxed
TrussStressFun
```

## Temperature

```@docs
TemperatureFun
cell_temperature
```

## Multi-load and trace estimation

```@docs
generate_scenarios
hutch_rand!
hadamard!
```

## Buckling helpers

```@docs
ElementKFun
AssembleKFun
apply_boundary_with_zerodiag!
apply_boundary_with_meandiag!
TrussElementKσFun
```

## Neural-network parametrization

```@docs
NeuralNetworkFun
TrainFunctionFun
PredictFunctionFun
AbstractMLModel
Coordinates
NNParams
getcentroids
```

## Multi-material

```@docs
MaterialInterpolationFun
MultiMaterialVariablesFun
element_densities
tounit
```

## Fixed element projection

`FixedElementProjectorFun` maps a reduced vector of free design variables to a
full element density vector, holding black (solid) and white (void) elements
fixed. Use `get_fixed_element_projector` to construct one from a problem or an
element count.

```@docs
FixedElementProjectorFun
```

```@docs
get_fixed_element_projector
```

```@docs
get_free_variables
```

```@docs
get_free_variable_count
```