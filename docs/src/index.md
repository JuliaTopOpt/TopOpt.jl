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

## Quick start

A minimal 2D SIMP example — minimize the compliance of a cantilever beam
subject to a volume-fraction constraint:

```julia
using TopOpt

# Problem setup (2D)
nels = (60, 20)
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)

# FEA solver with a power-law penalty
solver = FEASolver(DirectSolver, problem; xmin=1e-6, penalty=PowerPenalty(3.0))

# Differentiable objective and constraint
comp = Compliance(solver)
vol = Volume(solver; fraction=true)
filter = DensityFilter(solver; rmin=2.0)
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
