# `Functions`

This sub-module of `TopOpt` defines the differentiable building blocks used in
topology optimization formulations. The narrative descriptions and constructor
examples live in the [Functions](@ref) page; the entries below render the full
docstrings, including the fixed-element projection helpers.

```@meta
CurrentModule = TopOpt.Functions
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