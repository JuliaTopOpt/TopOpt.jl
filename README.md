# TopOpt

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- [![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/juliatopopt/TopOpt.jl) -->
[![Actions Status](https://github.com/juliatopopt/TopOpt.jl/workflows/CI/badge.svg)](https://github.com/juliatopopt/TopOpt.jl/actions)
[![codecov](https://codecov.io/gh/juliatopopt/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliatopopt/TopOpt.jl)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/dev)

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia). To learn more and see some examples, visit the [documentation](https://juliatopopt.github.io/TopOpt.jl/stable).

## Installation

To install `TopOpt.jl`, run:

```julia
using Pkg
pkg"add TopOpt"
```

To additionally load the visualization submodule of `TopOpt`, you will need to install `Makie.jl` using:

```julia
pkg"add Makie"
```

To load the package, use:

```julia
using TopOpt
```

and to optionally load the visualization sub-module as part of `TopOpt`, use:

```julia
using TopOpt, Makie
```

## Features available

All the following features are available in TopOpt.jl but the documentation is currently lacking! Feel free to open an issue to ask about how to use specific features.

- 2D and 3D truss topology optimization
- 2D and 3D continuum topology optimization
- Unstructured ground mesh
- Linear and quadratic triangle, quadrilateral, tetrahedron and hexahedron elements in ground mesh
- Fixed and non-design domain support
- Concentrated and distributed loads
- SIMP, RAMP, continuation SIMP/RAMP and BESO
- Compliance, volume and stress functions
- End-to-end topology optimization from INP file to VTK file
- Automatic differentiation of arbitrary Julia functions
- Method of moving asymptotes, NLopt, Ipopt and augmented Lagrangian algorithm for optimization
- Density and sensitivity filters
- Heaviside projection
- Handling load uncertainty in compliance-based topology optimization

## Contribute

We always welcome new contributors! Feel free to open an issue or reach out to us via email if you want to collaborate. There are plenty of things to do including beginner friendly tasks and research-oriented tasks. You can help us create the best topology optimization ecosystem in the world! Some beginner-friendly ideas you could be working on include:
1. Multi-material design parameterisation
2. Level set design parameterization
3. Lattice design parameterization
4. Neural network based design parameterization
5. Local volume constraints
6. Supporting rectilinear grids
7. Wrapping OpenLSTO_jll which is the precompiled binary for M2DOLab/OpenLSTO
8. Wrapping TopOpt_in_PETSc_jll which is the precompiled binary for topopt/TopOpt_in_PETSc
9. Reliability-based design optimization
10. Robust optimization
11. Stochastic optimization
