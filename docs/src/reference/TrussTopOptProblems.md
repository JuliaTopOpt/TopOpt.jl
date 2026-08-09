# `TrussTopOptProblems`

This sub-module of `TopOpt` defines truss topology optimization problems and
the IO helpers to load truss geometries from JSON or .geo files.

```@meta
CurrentModule = TopOpt.TrussTopOptProblems
```

## Problem types

```@docs
TrussProblem
```

## Material and cross-section containers

```@docs
TrussFEAMaterial
TrussFEACrossSec
```

## Grid

```@docs
TrussGrid
```

## Standard truss test problem

```@docs
PointLoadCantileverTruss
```

## IO

```@docs
load_truss_json
load_truss_geo
```