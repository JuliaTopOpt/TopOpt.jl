using TopOpt

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
solver = FEASolver(Displacement, Direct, problem)
obj = ComplianceObj(problem, solver, filtering = true, rmin = 3.0)
constr = VolConstr(problem, solver, 0.5)
optimizer = MMAOptimizer(obj, constr, MMA.MMA87(), ConjugateGradient(), maxiter=1000);
