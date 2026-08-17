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

### Optimization domains

- 2D and 3D continuum and truss domains ([continuum](reference/TopOptProblems.md),
  [truss](reference/TrussTopOptProblems.md))
- Unstructured ground meshes imported from Abaqus/FreeCAD `.inp` files
- Linear and quadratic triangle, quadrilateral, tetrahedron and hexahedron
  elements
- Fixed and non-design regions ([`FixedElementProjectorFun`](reference/Functions.md))
- Concentrated and distributed loads
- Multi-material design parametrization ([`MultiMaterialVariablesFun`](functions.md))

### High-level algorithms and penalty types

- Solid isotropic material with penalization (SIMP) and RAMP
- Continuation SIMP/RAMP
- Bi-directional evolutionary structural optimization (BESO) with soft-kill
- Topology optimization of binary structures (TOBS)
- Level-set topology optimization, in 2D and 3D
  ([`OpenLSTO`](reference/OpenLSTO.md))
- Rational, hyperbolic-sine and projected penalty functions
  ([`Utilities`](reference/Utilities.md))

### Differentiable functions

A library of differentiable building blocks — compliance, volume, stress,
displacement, temperature, buckling, and more — that can be chained into
arbitrary objectives and constraints and differentiated automatically with
Zygote. See [Functions](functions.md).

### Linear system solvers

- Direct sparse Cholesky/QR factorization
- Preconditioned conjugate gradient with matrix assembly
- Matrix-free preconditioned conjugate gradient
- Custom solver and preconditioner hooks ([`FEA`](reference/FEA.md))

### Optimization algorithms

Optimization is driven by [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl):

- Method of moving asymptotes (MMA)
- All the algorithms in NLopt, and Ipopt
- First-order augmented Lagrangian algorithm
- Nonlinear semidefinite programming for buckling-constrained optimization
- Surrogate-assisted and Bayesian optimization
- Integer nonlinear programming, and TOBS sequential integer linear programming

### Handling uncertainty

- Mean, variance, standard deviation and scalar-valued functions of per-scenario
  compliance under load uncertainty
- Reliability-based topology optimization

### Visualization and post-processing

- End-to-end workflow from INP import to VTK export
- Interactive visualization of designs and deformation using
  [Makie.jl](https://makie.juliaplots.org/stable/)
- Static browser-based visualization with camera controls for the rendered docs
- Interactive visualization using Dash apps and
  [DashVtk](https://github.com/JuliaTopOpt/DashVtk_Examples/tree/main/src/TopOptDemo)

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

## Problem types

TopOpt.jl provides several pre-defined problem types for common test cases:

### Structural problems (linear elasticity)
- **`PointLoadCantilever`** — Cantilever beam with a point load at the free end (2D or 3D)
- **`HalfMBB`** — Half Messerschmitt-Bölkow-Blohm beam, a standard benchmark (2D or 3D)
- **`LBeam`** — L-shaped beam (2D only)
- **`TieBeam`** — Tie-beam test problem (2D only)
- **`InpStiffness`** — Import from Abaqus/FreeCAD `.inp` files for arbitrary meshes

### Heat transfer problems
- **`HeatConductionProblem`** — Steady-state heat conduction with surface heat flux (2D or 3D)
- **`HeatTree`** — Tree-shaped heat conduction problem

### Truss problems
- **`TrussProblem`** — General truss topology optimization with stress/buckling constraints
- **`PointLoadCantileverTruss`** — Truss cantilever with point load

See the [TopOpt Tutorials](tutorials/index.md) section for complete, commented examples covering
stress-constrained optimization, heat sinks, multi-material design,
neural-network parametrization, trusses, and more.
