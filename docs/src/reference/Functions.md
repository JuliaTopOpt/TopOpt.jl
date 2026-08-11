# `Functions`

This sub-module of `TopOpt` defines the differentiable building blocks used in
topology optimization formulations. The narrative descriptions and constructor
examples live in the [Functions](@ref) page; the entries below render the full
docstrings.

```@meta
CurrentModule = TopOpt.Functions
```

## Compliance and volume

```@docs
Compliance
Volume
ThermalCompliance
MeanCompliance
BlockCompliance
```

## Displacement and stress

```@docs
Displacement
StressTensor
ElementStressTensor
von_mises_stress_function
TrussStress
```

## Buckling helpers

```@docs
ElementK
AssembleK
apply_boundary_with_zerodiag!
apply_boundary_with_meandiag!
TrussElementKσ
```

## Neural-network parametrization

```@docs
NeuralNetwork
TrainFunction
PredictFunction
```

## Multi-material

```@docs
MaterialInterpolation
MultiMaterialVariables
element_densities
tounit
```

## Fixed element projection

`FixedElementProjector` maps a reduced vector of free design variables to a
full element density vector, holding black (solid) and white (void) elements
fixed. Use `get_fixed_element_projector` to construct one from a problem or an
element count.

```@docs
FixedElementProjector
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