# TopOpt.jl Documentation

## Introduction

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).

!!! note

    `TopOpt` is still under development. If you find a bug, or have
    ideas for improvements, feel free to open an issue or make a
    pull request on the [`TopOpt` GitHub page](https://github.com/mohamed82008/TopOpt.jl).

## Installation

In Julia v1.0+ you can install packages using Julia's package manager as follows:

```julia
using Pkg
pkg"add https://github.com/yijiangh/Tensors.jl.git#master"
pkg"add https://github.com/yijiangh/JuAFEM.jl.git"
pkg"add https://github.com/mohamed82008/VTKDataTypes.jl#master"
pkg"add https://github.com/mohamed82008/Nonconvex.jl#master"
pkg"add https://github.com/mohamed82008/TopOpt.jl#master"
```

which will track the `master` branch of the package. To additionally load the visualization submodule of `TopOpt`, you will need to install `Makie.jl` using:

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
