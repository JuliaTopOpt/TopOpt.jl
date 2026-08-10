# `Algorithms`

This sub-module of `TopOpt` defines topology-optimization-specific algorithms
(`BESO`, `GESO`). General-purpose nonlinear optimization (SIMP via `MMA87`/
`MMA02`, `TOBSAlg`, `IpoptAlg`, etc.) is handled through
[`Nonconvex.jl`](https://github.com/JuliaNonconvex/Nonconvex.jl), which
`TopOpt` re-exports. See the [Examples](@ref Examples) for usage.

```@meta
CurrentModule = TopOpt.Algorithms
```

## Abstract type

```@docs
TopOptAlgorithm
```

## BESO

```@docs
BESO
```

## GESO

```@docs
GESO
```

