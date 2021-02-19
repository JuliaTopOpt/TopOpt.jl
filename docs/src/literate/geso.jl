using TopOpt

nels = (10, 10)
problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
solver = FEASolver(Displacement, Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(3.0))
comp = Compliance(problem, solver)
volfrac = Volume(problem, solver)
sensfilter = SensFilter(solver, rmin = 4.0)
geso = GESO(comp, IneqConstraint(volfrac, 0.5), sensfilter)
x0 = ones(length(solver.vars))
resul = geso(x0)
