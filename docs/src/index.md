# TopOpt.jl Documentation

## Introduction

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).
It supports both **structural mechanics** (linear elasticity) and **heat
transfer** (heat conduction) problems, on either continuum or truss ground
meshes, in 2D and 3D. Optimization is carried out through
[`Nonconvex.jl`](https://github.com/JuliaNonconvex/Nonconvex.jl), which provides
MMA, IPOPT, TOBS, Juniper, and other solvers.

!!! note

    `TopOpt` is still under development. If you find a bug, or have
    ideas for improvements, feel free to open an issue or make a
    pull request on the [`TopOpt` GitHub page](https://github.com/JuliaTopOpt/TopOpt.jl).

## Installation

To install `TopOpt.jl`, run:

```julia
using Pkg
pkg"add TopOpt"
```

To additionally load the visualization submodule of `TopOpt`, you will need to install `Makie.jl` using:

```julia
pkg"add Makie, GLMakie"
```

To load the package, use:

```julia
using TopOpt
```

and to optionally load the visualization sub-module as part of `TopOpt`, use:

```julia
using TopOpt, Makie, GLMakie
```

## Features

TopOpt.jl is organized around a small set of high-level building blocks:
problem types define the design domain and physics, differentiable functions
define objectives and constraints, filters regularize the design, and
algorithms drive the optimization. Every feature links to its reference page.

### Optimization domains

- **Continuum structural** (2D/3D linear elasticity): `PointLoadCantilever`,
  `HalfMBB`, `LBeam`, `TieBeam`, and arbitrary meshes imported from
  Abaqus/FreeCAD `.inp` files via `InpStiffness`
  ([`TopOptProblems`](reference/TopOptProblems.md))
- **Truss** (2D/3D): `TrussProblem` and `PointLoadCantileverTruss`, with
  stress and buckling constraints
  ([`TrussTopOptProblems`](reference/TrussTopOptProblems.md))
- **Heat transfer** (2D/3D): `HeatConductionProblem` and `HeatTree`
  ([`TopOptProblems`](reference/TopOptProblems.md))
- Linear and quadratic triangle, quadrilateral, tetrahedron and hexahedron
  elements
- Concentrated and distributed loads, multi-load cases (`MultiLoad`), and
  fixed/non-design regions ([`FixedElementProjectorFun`](reference/Functions.md))

### Optimization methods

- **Density-based**: SIMP and RAMP, with continuation
- **BESO** and **GESO** evolutionary algorithms ([`Algorithms`](reference/Algorithms.md))
- **TOBS**: binary topology optimization via sequential integer programming
- **Level-set**: 2D and 3D level-set topology optimization, with hole
  nucleation, stress minimization, and marching-cubes boundary discretization
  ([`OpenLSTO`](reference/OpenLSTO.md))
- Gradient-based optimization through
  [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl): MMA, Ipopt,
  NLopt, augmented Lagrangian, nonlinear SDP (buckling), and Juniper
  (mixed-integer)

### Differentiable functions

Composable, Zygote-differentiable building blocks for objectives and
constraints ([guide](functions.md), [reference](reference/Functions.md)):

- **Objectives**: `ComplianceFun`, `ThermalComplianceFun`, `VolumeFun`,
  `MeanComplianceFun`, `BlockComplianceFun`
- **Responses**: `DisplacementFun`, `TemperatureFun`, `StressTensorFun`,
  `von_mises_stress_function`, `epsilon_relaxed`, `TrussStressFun`
- **Buckling**: `ElementKFun`, `AssembleKFun`, `TrussElementKσFun`
- **Parametrization**: `NeuralNetworkFun`, `MaterialInterpolationFun`,
  `MultiMaterialVariablesFun`, `FixedElementProjectorFun`

### Filters and penalties

- `DensityFilterFun`, `SensFilterFun`, `ProjectedDensityFilterFun`
  ([`CheqFilters`](reference/CheqFilters.md))
- Power, rational and hyperbolic-sine penalties, and Heaviside/sigmoid
  projections ([`Utilities`](reference/Utilities.md))

### Linear system solvers

- Direct sparse Cholesky/QR factorization (`DirectSolver`)
- Preconditioned conjugate gradient, with assembly and matrix-free
  (`CGAssemblySolver`, `CGMatrixFreeSolver`)
- Custom solver and preconditioner hooks ([`FEA`](reference/FEA.md))

### Visualization and input/output

- Interactive and static (browser, camera-control) visualization with
  [Makie.jl](https://makie.juliaplots.org/stable/)
- VTK export (`save_mesh`) and Abaqus/FreeCAD `.inp` import (`InpStiffness`)
- STL export of 3D level-set designs (`write_stl`)

### Handling uncertainty

- Mean and per-scenario compliance under load uncertainty
  (`MeanComplianceFun`, `BlockComplianceFun`)
- Reliability-based topology optimization

## Quick start

A minimal 2D SIMP example — minimize the compliance of a cantilever beam
subject to a volume-fraction constraint:

```julia
using TopOpt

# Problem setup (2D)
nels = (60, 20)
problem = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)

# FEA solver with a power-law penalty
solver = FEASolver(DirectSolver, problem; xmin=1e-6, penalty=PowerPenaltyFun(3.0))

# Differentiable objective and constraint
comp = ComplianceFun(solver)
vol = VolumeFun(solver; fraction=true)
filter = DensityFilterFun(solver; rmin=2.0)
obj = x -> comp(filter(PseudoDensities(x)))
constr = x -> vol(filter(PseudoDensities(x))) - 0.3

# Optimize with MMA
x0 = fill(0.3, length(solver.vars))
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
result = optimize(model, MMA87(), x0)

# Visualize the result (requires Makie)
using Makie, GLMakie
fig = visualize(problem; topology=result.minimizer)
Makie.display(fig)
```

For an interactive camera-control app suitable for static HTML export, use
`visualize(problem; static=true, topology=result.minimizer)` with WGLMakie and
initialize Bonito with `Bonito.Page(exportable=true, offline=true)` first. This
returns a `Bonito.App`.

## Where to go next

The [Tutorials](tutorials/index.md) walk through complete, commented examples
— compliance and volume minimization, stress-constrained and
buckling-constrained optimization, heat sinks, multi-material and
neural-network parametrization, trusses, and level-set optimization. The
sidebar lists the full API reference, grouped by module.
