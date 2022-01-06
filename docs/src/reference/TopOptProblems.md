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

### Test problems

The following types are all concrete subtypes of `StiffnessTopOptProblem`. `PointLoadCantilever` is a cantilever beam problem with a point load as shown below. `HalfMBB` is the half Messerschmitt-Bölkow-Blohm (MBB) beam problem commonly used in topology optimization literature. `LBeam` and `TieBeam` are the common L-beam and tie-beam test problem used in topology optimization literature. The `PointLoadCantilever` and `HalfMBB` problems can be either 2D or 3D depending on the type of the inputs to the constructor. If the number of elements and sizes of elements are 2-tuples, the problem constructed will be 2D. And if they are 3-tuples, the problem constructed will be 3D. For the 3D versions, the point loads are applied at approximately the mid-depth point. The `TieBeam` and `LBeam` problems are always 2D.

```@docs
PointLoadCantilever
PointLoadCantilever(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force) where {dim, CellType}
```

```@docs
HalfMBB
HalfMBB(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force) where {dim, CellType}
```

```@docs
LBeam
LBeam(::Type{Val{CellType}}, ::Type{T}=Float64; length = 100, height = 100, upperslab = 50, lowerslab = 50, E = 1.0, ν = 0.3, force = 1.0) where {T, CellType}
```

```@docs
TieBeam
TieBeam(::Type{Val{CellType}}, ::Type{T} = Float64, refine = 1, force = T(1); E = T(1), ν = T(0.3)) where {T, CellType}
```

### Reading INP Files

In `TopOpt.jl`, you can import a `.inp` file to an instance of the problem struct `InpStiffness`. This can be used to construct problems with arbitrary unstructured ground meshes, complex boundary condition domains and load specifications. The `.inp` file can be exported from a number of common finite element software such as: FreeCAD or ABAQUS.

```@docs
InpStiffness
InpStiffness(filepath_with_ext::AbstractString; keep_load_cells = false)
```

```@docs
IO.INP.Parser.InpContent
```

## Grids

Grid types are defined in `TopOptProblems` because a number of topology optimization problems share the same underlying grid but apply the loads and boundary conditions at different locations. For example, the `PointLoadCantilever` and `HalfMBB` problems use the same rectilinear grid type, `RectilinearGrid`, under the hood. The `LBeam` problem uses the `LGrid` function under the hood to construct an L-shaped `Ferrite.Grid`. New problem types can be defined using the same grids but different loads or boundary conditions.

```@docs
RectilinearGrid
RectilinearGrid(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim,T}) where {dim, T, CellType}
```

```@docs
LGrid
```

## Finite element backend

Currently, `TopOpt` uses [`Ferrite.jl`](https://github.com/KristofferC/Ferrite.jl) for FEA-related modeling. 
This means that all the problems above are described in the language and types of `Ferrite`.

## Matrices and vectors

### `ElementFEAInfo`

```@docs
ElementFEAInfo
ElementFEAInfo(sp, quad_order, ::Type{Val{mat_type}}) where {mat_type}
```

### `GlobalFEAInfo`

```@docs
GlobalFEAInfo
GlobalFEAInfo(::Type{T}=Float64) where {T}
GlobalFEAInfo(sp::StiffnessTopOptProblem)
```
