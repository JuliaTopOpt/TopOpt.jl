# TopOpt.jl Documentation

## Introduction

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).

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
