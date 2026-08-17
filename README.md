
<p align="center">
  <img src="assets/logo.png" width="400" alt="TopOpt.jl logo">
</p>

# TopOpt.jl

[![Actions Status](https://github.com/JuliaTopOpt/TopOpt.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/juliatopopt/TopOpt.jl/actions)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/stable)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/dev)
[![codecov](https://codecov.io/gh/juliatopopt/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliatopopt/TopOpt.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-Blue-4473c9.svg)](https://github.com/invenia/BlueStyle)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia). It supports structural (linear elasticity) and heat-transfer problems on continuum and truss ground meshes in 2D and 3D, with automatic differentiation through every objective and constraint.

![gif1](https://user-images.githubusercontent.com/19524993/138464511-2685f3fe-e7c5-482e-8b06-43ab0fb82990.gif)
![gif2](https://user-images.githubusercontent.com/19524993/138464828-88f0ffcb-01f7-43b7-8d17-f5d201e95aa3.gif)
![gif3](https://user-images.githubusercontent.com/19524993/138464845-d0b289b7-0fe9-4408-be57-fe697b5d671e.gif)
![gif4](https://user-images.githubusercontent.com/19524993/167059067-f08502a8-c62d-4d62-a2df-e132efc5e25c.gif)

## Installation

See the [installation section of the documentation](https://juliatopopt.github.io/TopOpt.jl/stable/#installation)
([dev](https://juliatopopt.github.io/TopOpt.jl/dev/#installation)).

## Features

The full list of features is maintained in one place, the
[documentation](https://juliatopopt.github.io/TopOpt.jl/stable/#features)
([dev](https://juliatopopt.github.io/TopOpt.jl/dev/#features)).

## Tutorials

Executable, commented tutorials are hosted on the documentation site:
[TopOpt.jl Tutorials](https://juliatopopt.github.io/TopOpt.jl/stable/tutorials/)
([dev](https://juliatopopt.github.io/TopOpt.jl/dev/tutorials/)).

## Citation

If you use TopOpt.jl in your research, please cite the following journal paper
and conference publications.

- [Adaptive continuation solid isotropic material with penalization for volume constrained compliance minimization](https://doi.org/10/gg55cc)

```bibtex
@article{TarekRay2020,
  title={Adaptive continuation solid isotropic material with penalization for volume constrained compliance minimization},
  author={Tarek, Mohamed and Ray, Tapabrata},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={363},
  pages={112880},
  year={2020},
  doi={10/gg55cc}
}
```

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

A standard citation file is provided as [`CITATION.bib`](CITATION.bib).

## Contribute

We welcome new contributors! Please see the
[open issues](https://github.com/JuliaTopOpt/TopOpt.jl/issues) for beginner-friendly
and research-oriented tasks, or open a new issue to discuss your idea.

## Questions?

If you have any questions, join us on the #topopt channel in the [Julia slack](https://julialang.org/slack/), open an issue or shoot us an email.
