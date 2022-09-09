using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (30, 10, 10)
problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0, 1.0), E, v, f);

V = 0.3 # volume fraction
xmin = 1e-6 # minimum density
rmin = 2.0; # density filter radius

penalty = TopOpt.PowerPenalty(3.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

comp = TopOpt.Compliance(solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(PseudoDensities(x)))

volfrac = TopOpt.Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V

x0 = fill(V, length(solver.vars))
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()
options = MMAOptions(; maxiter=3000, tol=Nonconvex.Tolerance(; x=1e-3, f=1e-3, kkt=0.001))
convcriteria = Nonconvex.KKTCriteria()
r = optimize(model, alg, x0; options, convcriteria)

@show obj(r.minimizer)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

