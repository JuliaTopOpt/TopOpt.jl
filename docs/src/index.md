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

A minimal SIMP example — minimize the compliance of a 3-D cantilever beam
subject to a volume-fraction constraint:

```julia
using TopOpt

# Problem setup
problem = PointLoadCantilever(Val{:Linear}, (30, 10, 10), (1.0, 1.0, 1.0), 1.0, 0.3, 1.0)

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
```

See the [Examples](@ref) section for complete, commented examples covering
stress-constrained optimization, heat sinks, multi-material design,
neural-network parametrization, trusses, and more.
