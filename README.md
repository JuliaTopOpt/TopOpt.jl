# TopOpt

## Example

```julia
# Setup

]add https://github.com/mohamed82008/TopOpt.jl#master
]add Makie

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

# Define a finite element solver

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
    penalty = penalty)

# Define compliance objective

obj = Objective(ComplianceFunction(problem, solver, filtering = true,
    rmin = 4.0, tracing = true, logarithm = false))

# Define volume constraint

constr = Constraint(VolumeFunction(problem, solver), V)

# Define subproblem optimizer

mma_options = options = MMA.Options(maxiter = 3000, 
    tol = MMA.Tolerances(kkttol = 0.001))
convcriteria = MMA.KKTCriteria()
optimizer = MMAOptimizer{CPU}(obj, constr, MMA.MMA87(),
    ConjugateGradient(), options = mma_options,
    convcriteria = convcriteria)

# Define SIMP optimizer

simp = SIMP(optimizer, penalty.p)

# Solve

x0 = fill(1.0, length(solver.vars));
result = simp(x0)

# Visualize the result using Makie.jl

glmesh, color = GLMesh(problem, result.topology)
mesh(glmesh, color=color)
```
