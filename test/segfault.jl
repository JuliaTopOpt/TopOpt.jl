using TopOpt

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
solver = FEASolver(Displacement, Direct, problem)
obj = Compliance(problem, solver, filtering = true, rmin = 3.0)
constr = Constraint(Volume(problem, solver, 0.5))
optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), maxiter=1000);
