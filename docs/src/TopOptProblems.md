# `TopOptProblems`

This sub-module of `TopOpt` defines a number of standard topology optimization problems for the convenient testing of algorithms.

```@meta
CurrentModule = TopOpt.TopOptProblems
```

## Problem types

### Abstract type

`StiffnessTopOptProblem` is an abstract type that a number of linear elasticity, quasi-static, topology optimization problems subtype.

```@docs
StiffnessTopOptProblem
```

### Concrete types

The following types are all concrete subtypes of `StiffnessTopOptProblem`. `PointLoadCantilever` is a cantilever beam problem with a point load as shown below. `HalfMBB` is the half Messerschmitt-Bölkow-Blohm (MBB) beam problem commonly used in topology optimization literature. `LBeam` and `TieBeam` are the common L-beam and tie-beam test problem used in topology optimization literature. The `PointLoadCantilever` and `HalfMBB` problems can be either 2D or 3D depending on the type of the inputs to the constructor. If the number of elements and sizes of elements are 2-tuples, the problem constructed will be 2D. And if they are 3-tuples, the problem constructed will be 3D. For the 3D versions, the point loads are applied at approximately the mid-depth point. The `TieBeam` and `LBeam` problems are always 2D.

Finally, `InpStiffness` is a problem type that is instantiated by importing an `.inp` file. This can be used to represent an arbitrary unstructured mesh with complex boundary condition domains and load specification. The `.inp` file can be exported from a number of common finite element software such as: FreeCAD or ABAQUS.

```@docs
PointLoadCantilever
HalfMBB
LBeam
TieBeam
InpStiffness
```

## Grids

Grid types are defined in `TopOptProblems` because a number of topology optimization problems share the same underlying grid but apply the loads and boundary conditions at different locations. For example, the `PointLoadCantilever` and `HalfMBB` problems use the same rectilinear grid, `RectilinearGrid`, under the hood. The `LBeam` problem uses the `LGrid` grid type under the hood. New problem types can be defined using the same grid types but different loads or boundary conditions.

```@docs
RectilinearGrid
LGrid
```

## Finite element backend

Currently, `TopOpt` uses a [forked version](https://github.com/mohamed82008/JuAFEM.jl) of [`JuAFEM.jl`](https://github.com/KristofferC/JuAFEM.jl). This means that all the problems above are described in the language and types of `JuAFEM`.

The fork is needed for GPU support but the main package should also work on the CPU. The changes in the fork should make it back to the main repo at some point.

## Matrices and vectors

### `ElementFEAInfo`

```@docs
ElementFEAInfo
```

### `GlobalFEAInfo`

```@docs
GlobalFEAInfo
```

