# TopOpt

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- [![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/TopOpt.jl) -->
[![Actions Status](https://github.com/mohamed82008/TopOpt.jl/workflows/CI/badge.svg)](https://github.com/mohamed82008/TopOpt.jl/actions)
[![codecov](https://codecov.io/gh/mohamed82008/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mohamed82008/TopOpt.jl)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://mohamed82008.github.io/TopOpt.jl/dev)

`TopOpt` is a topology optimization package written in [Julia](https://github.com/JuliaLang/julia).

## Installation

To install `TopOpt.jl`, you can either (1) add it to an existing Julia environment or (2) clone and use its shipped environment.
The second option is recommended for new users who simply wants to try this package out.

### Adding `TopOpt` to an existing Julia environment

In Julia v1.0+ you can install packages from the Pkg REPL (press `]` in the Julia
REPL to enter `pkg>` mode):

<!-- pkg> add https://github.com/KristofferC/JuAFEM.jl.git -->
```julia
pkg> add https://github.com/yijiangh/Tensors.jl.git#master
pkg> add https://github.com/yijiangh/JuAFEM.jl.git
pkg> add https://github.com/mohamed82008/VTKDataTypes.jl#master
pkg> add https://github.com/mohamed82008/KissThreading.jl#master
pkg> add https://github.com/mohamed82008/TopOpt.jl#master
```

which will track the `master` branch of the package.

### Using `TopOpt`'s shipped environment

Clone the repo by:
```
git clone https://github.com/mohamed82008/TopOpt.jl.git
cd TopOpt.jl
```

Then issue the following in your commandline to activate and build the environment: 

```bash
# launch Julia with the project at current dir on startup
julia --project=@.
julia> using Pkg; Pkg.build(); 
```

These will fetch all the required dependencies and precompile them.
You can learn more about Julia's package environment system [here](https://julialang.github.io/Pkg.jl/v1/environments/).

## Getting Started

To load the package, use

```julia
using TopOpt
```

See [documentation](https://mohamed82008.github.io/TopOpt.jl/dev/examples/point_load_cantilever/) for example usages.