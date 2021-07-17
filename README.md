# TopOpt

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- [![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/juliatopopt/TopOpt.jl) -->
[![Actions Status](https://github.com/juliatopopt/TopOpt.jl/workflows/CI/badge.svg)](https://github.com/juliatopopt/TopOpt.jl/actions)
[![codecov](https://codecov.io/gh/juliatopopt/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliatopopt/TopOpt.jl)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://juliatopopt.github.io/TopOpt.jl/dev)

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).

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
