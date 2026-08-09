# `Utilities`

This sub-module of `TopOpt` defines penalty functions, projections, and
low-level helpers shared across the package. The penalty and projection types
control how raw design variables are mapped to physical densities before they
enter the stiffness assembly.

```@meta
CurrentModule = TopOpt.Utilities
```

## Penalties

Penalties (SIMP-style) push intermediate densities toward 0 or 1 so that the
optimization converges to a near-binary design.

```@docs
AbstractPenalty
PowerPenalty
RationalPenalty
SinhPenalty
ProjectedPenalty
```

## Projections

Projections are smooth approximations of a step function, used to sharpen the
design. They can be applied standalone or composed with a penalty via
`ProjectedPenalty`.

```@docs
AbstractProjection
HeavisideProjection
SigmoidProjection
```

## Penalty accessors

```@docs
getpenalty
getprevpenalty
setpenalty!
```

## Design-density helper

```@docs
density
```