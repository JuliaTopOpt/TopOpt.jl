# TopOpt

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/TopOpt.jl)
[![codecov](https://codecov.io/gh/mohamed82008/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mohamed82008/TopOpt.jl)

`TopOpt` is a a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).

## Documentation

[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://mohamed82008.github.io/TopOpt.jl/dev)

## Installation

In Julia v1.0+ you can install JuAFEM from the Pkg REPL (press `]` in the Julia
REPL to enter `pkg>` mode):

```
pkg> add https://github.com/KristofferC/JuAFEM.jl.git
pkg> add https://github.com/mohamed82008/VTKDataTypes.jl#master
pkg> add https://github.com/mohamed82008/KissThreading.jl#master
pkg> add https://github.com/mohamed82008/TopOpt.jl#master
```

which will track the `master` branch of the package.

To load the package, use

```julia
using TopOpt
```