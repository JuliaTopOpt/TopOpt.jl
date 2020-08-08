[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/YingboMa/SafeTestsets.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/TopOpt.jl)
[![codecov](https://codecov.io/gh/mohamed82008/TopOpt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mohamed82008/TopOpt.jl)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://mohamed82008.github.io/TopOpt.jl/dev)

# TopOpt

## Example

```julia
# Setup

using Pkg

pkg"add https://github.com/mohamed82008/JuAFEM.jl#master"
pkg"add https://github.com/mohamed82008/VTKDataTypes.jl#master"
pkg"add https://github.com/mohamed82008/KissThreading.jl#master"
pkg"add https://github.com/mohamed82008/TopOpt.jl#master"
pkg"add Makie"

# Load the package

using TopOpt, Makie

# Define the problem

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(Val{:Linear}, (40, 20, 20), (1.0, 1.0, 1.0), E, v, f)
# problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f)
# problem = PointLoadCantilever(Val{:Quadratic}, (160, 40), (1.0, 1.0), E, v, f)
# problem = LBeam(Val{:Linear}, Float64)

# Parameter settings

V = 0.3 # volume fraction
xmin = 0.001 # minimum density
rmin = 4.0 # density filter radius

# Define a finite element solver

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
    penalty = penalty)

# Define compliance objective

obj = Objective(TopOpt.Compliance(problem, solver, filterT = DensityFilter,
    rmin = rmin, tracing = true, logarithm = false))

# Define volume constraint

constr = Constraint(TopOpt.Volume(problem, solver, filterT = DensityFilter, rmin = rmin), V)

# Define subproblem optimizer

mma_options = options = MMA.Options(maxiter = 3000, 
    tol = MMA.Tolerances(kkttol = 0.001))
convcriteria = MMA.KKTCriteria()
optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
    ConjugateGradient(), options = mma_options,
    convcriteria = convcriteria)

# Define SIMP optimizer

simp = SIMP(optimizer, penalty.p)

# Solve

x0 = fill(1.0, length(solver.vars));
result = simp(x0)

# Visualize the result using Makie.jl

glmesh = GLMesh(problem, result.topology)
mesh(glmesh)
```
