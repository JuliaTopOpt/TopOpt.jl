# TopOpt

[![Actions Status](https://github.com/juliatopopt/TopOpt.jl/workflows/CI/badge.svg)](https://github.com/juliatopopt/TopOpt.jl/actions)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/dev)
[![codecov](https://codecov.io/gh/juliatopopt/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliatopopt/TopOpt.jl)

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

![gif1](https://user-images.githubusercontent.com/19524993/138464511-2685f3fe-e7c5-482e-8b06-43ab0fb82990.gif)
![gif2](https://user-images.githubusercontent.com/19524993/138464828-88f0ffcb-01f7-43b7-8d17-f5d201e95aa3.gif)
![gif3](https://user-images.githubusercontent.com/19524993/138464845-d0b289b7-0fe9-4408-be57-fe697b5d671e.gif)
![gif4](https://user-images.githubusercontent.com/19524993/167059067-f08502a8-c62d-4d62-a2df-e132efc5e25c.gif)

## Features available

All the following features are available in TopOpt.jl but the documentation is currently lacking! Feel free to open an issue to ask about how to use specific features.

### Optimizaton domains

- 2D and 3D truss domains
- 2D and 3D continuum domains
- Unstructured ground mesh
- Linear and quadratic triangle, quadrilateral, tetrahedron and hexahedron elements in ground mesh
- Fixed and non-design domain support
- Concentrated and distributed loads

### High level algorithms and penalty types

The following high level topology optimization algorithms and penalty types are available.

- Solid isotropic material with penalization (SIMP)
- Rational approximation of material properties (RAMP)
- Continuation SIMP/RAMP
- Bi-directional evolutionary structural optimization (BESO) with soft-kill
- Topology optimization of binary structures (TOBS)

### Differentiable functions

There are numerous functions in `TopOpt.jl` that are defined in a differentiable way and you can use them in the objectives or constraints in topology optimization formulations. In `TopOpt.jl`, you can build arbitrarily complex objective and constraint functions using these differentiable functions as building blocks or lego pieces chaining them in any arbitrary way. The gradient and jacobian of the aggregate Julia function defined is then obtained using [automatic differentiation](https://www.youtube.com/watch?v=UqymrMG-Qi4). For a detailed account of the functions available, see the relevant section in the [documentation](https://juliatopopt.github.io/TopOpt.jl/dev/functions/).

### Linear system solvers

- Direct sparse Cholesky decomposition based linear system solver
- Preconditioned conjugate gradient method with matrix assembly
- Matrix-free preconditioned conjugate gradient method

### Optimization algorithms

We use [Nonconvex.jl](https://github.com/JuliaNonconvex/Nonconvex.jl) for the optimization problem definition and solving. The following algorithms are all available using `Nonconvex.jl`.

- Method of moving asymptotes
- All the algorithms in NLopt
- Ipopt
- First order augmented Lagrangian algorithm
- Nonlinear semidefinite programming for buckling constrained optimization
- Basic surrogate assisted optimization and Bayesian optimization
- Integer nonlinear programming (design variables guaranteed to be integer)
- Sequential integer linear programming in the topology optimization for binary structures (TOBS) algorithm

### Visualization and post-processing

- End-to-end topology optimization from INP file to VTK file
- Interactive visualization of designs and deformation using [Makie.jl](https://makie.juliaplots.org/stable/)
- Interactive visualization of designs using Dash apps and [DashVtk](https://github.com/JuliaTopOpt/DashVtk_Examples/tree/main/src/TopOptDemo)

### Handling uncertainty
- Handling load uncertainty in compliance-based topology optimization

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
