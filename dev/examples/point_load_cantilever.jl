using TopOpt, Makie
import GeometryBasics

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (12, 6, 6) # change to (40, 20, 20) for a more high-res result
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0, 1.0), E, v, f);

V = 0.3 # volume fraction
xmin = 0.001 # minimum density
rmin = 4.0; # density filter radius

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
    penalty = penalty);

obj = Objective(TopOpt.Compliance(problem, solver, filterT = DensityFilter,
    rmin = rmin, tracing = true, logarithm = false));

constr = Constraint(TopOpt.Volume(problem, solver, filterT = DensityFilter, rmin = rmin), V);

mma_options = options = MMA.Options(maxiter = 3000,
    tol = MMA.Tolerances(kkttol = 0.001))
convcriteria = MMA.KKTCriteria()
optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
    ConjugateGradient(), options = mma_options,
    convcriteria = convcriteria);

simp = SIMP(optimizer, penalty.p);

x0 = fill(1.0, length(solver.vars))
result = simp(x0);

result_mesh = GeometryBasics.Mesh(problem, result.topology);

mesh(result_mesh);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

