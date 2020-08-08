using TopOpt, TopOptProblems, CuArrays, LinearAlgebra

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
maxtol = 0.01 # maximum tolerance
mintol = 0.0001 # minimum tolerance
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
reuse = true # adaptive penalty flag
filterT = DensityFilter
solver = FEASolver(Displacement, CG, MatrixFree, problem, xmin = xmin,
	penalty = TopOpt.PowerPenalty(1.0))	
cusolver = cu(solver)
cuobj = ComplianceFunction(problem, cusolver, filterT = DensityFilter,
rmin = 3.0, tracing = true, logarithm = false)

operator = TopOpt.buildoperator(cusolver)
x = CuArray(ones(length(cusolver.f)))
mul!(x, operator, cusolver.f)
