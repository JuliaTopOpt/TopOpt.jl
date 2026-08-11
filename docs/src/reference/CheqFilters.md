# `CheqFilters`

This sub-module of `TopOpt` defines chequerboard filters that suppress
mesh-dependent checkerboard patterns in the optimized design. There are two
families: **sensitivity filters** (smooth the gradient) and **density filters**
(smooth the design variables).

```@meta
CurrentModule = TopOpt.CheqFilters
```

## Abstract types

```@docs
AbstractCheqFilter
AbstractSensFilter
AbstractDensityFilter
```

## Sensitivity filter

```@docs
SensFilter
```

## Density filters

```@docs
DensityFilter
ProjectedDensityFilter
```