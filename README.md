# TopOpt

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- [![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/juliatopopt/TopOpt.jl) -->
[![Actions Status](https://github.com/juliatopopt/TopOpt.jl/workflows/CI/badge.svg)](https://github.com/juliatopopt/TopOpt.jl/actions)
<!-- [![codecov](https://codecov.io/gh/juliatopopt/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliatopopt/TopOpt.jl) -->
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/dev)

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia). To learn more and see some examples, visit the [documentation](https://juliatopopt.github.io/TopOpt.jl/stable). Numerous examples can also be found in the `test/examples` and `test/wcsmo14` directories.

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
- Buckling constrained truss optimization
- End-to-end topology optimization from INP file to VTK file
- Interactive visualization of designs and deformation
- Automatic differentiation of arbitrary Julia functions
- Method of moving asymptotes, NLopt, Ipopt and augmented Lagrangian algorithm for optimization
- Density and sensitivity filters
- Heaviside projection
- Handling load uncertainty in compliance-based topology optimization
- Neural network representation of designs
- Integer nonlinear topology optimization for truss and continuum problems (design variables guaranteed to be integer)
- Topology optimization of binary structures (TOBS) using integer linear Programming

![gif1](https://user-images.githubusercontent.com/19524993/138464511-2685f3fe-e7c5-482e-8b06-43ab0fb82990.gif)
![gif2](https://user-images.githubusercontent.com/19524993/138464828-88f0ffcb-01f7-43b7-8d17-f5d201e95aa3.gif)
![gif3](https://user-images.githubusercontent.com/19524993/138464845-d0b289b7-0fe9-4408-be57-fe697b5d671e.gif)


## Citation

To cite this package, you can cite the following 2 conference publications.

- TopOpt.jl: An efficient and high-performance package for topology optimization of continuum structures in the Julia programming language

```bibtex
@inproceedings{tarek2019topoptjl,
  title={TopOpt.jl: An efficient and high-performance package for topology optimization of continuum structures in the Julia programming language},
  author={Tarek, Mohamed},
  booktitle={Proceedings of the 13th World Congress of Structural and Multidisciplinary Optimization},
  year={2019}
}
```

- [TopOpt.jl: Truss and Continuum Topology Optimization, Interactive Visualization, Automatic Differentiation and More](https://web.mit.edu/yijiangh/www/papers/topopt_jl_WCSMO2021.pdf)

```bibtex
@inproceedings{huang2021topoptjl,
  title={TopOpt.jl: Truss and Continuum Topology Optimization, Interactive Visualization, Automatic Differentiation and More},
  author={Huang, Yijiang and Tarek, Mohamed},
  booktitle={Proceedings of the 14th World Congress of Structural and Multidisciplinary Optimization},
  year={2021}
}
```

## Contribute

We always welcome new contributors! Feel free to open an issue or reach out to us via email if you want to collaborate. There are plenty of things to do including beginner friendly tasks and research-oriented tasks. You can help us create the best topology optimization ecosystem in the world! Some beginner-friendly ideas you could be working on include:
- Multi-material design parameterisation
- Level set design parameterization
- Lattice design parameterization
- Local volume constraints
- Supporting rectilinear grids
- Wrapping OpenLSTO_jll which is the precompiled binary for M2DOLab/OpenLSTO
- Wrapping TopOpt_in_PETSc_jll which is the precompiled binary for topopt/TopOpt_in_PETSc
- Reliability-based design optimization
- Robust optimization
- Stochastic optimization


## Questions?

If you have any questions, join us on on the #topopt channel in the [Julia slack](https://julialang.org/slack/), open an issue or shoot us an email.
