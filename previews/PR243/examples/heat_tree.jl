using TopOpt

nels = (80, 40)
k = 1.0
q = 1.0
problem = HeatTree(Val{:Linear}, nels, (1.0, 1.0), k; q=q)

V = 0.3      # volume fraction
xmin = 1e-3  # minimum density
rmin = 2.0

x0 = fill(V, TopOpt.getncells(problem))
solver = FEASolver(DirectSolver, problem; xmin=xmin, penalty=PowerPenalty(3.0))
comp = ThermalCompliance(solver)
filter = DensityFilter(solver; rmin=rmin)
obj = x -> comp(filter(PseudoDensities(x)))

volfrac = Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()

options = MMAOptions(; tol=Tolerance(; kkt=1e-4), maxiter=200)
res = optimize(model, alg, x0; options)

@show obj(res.minimizer)
@show constr(res.minimizer)

using Makie
using CairoMakie

fig = visualize(problem; topology=res.minimizer)
Makie.display(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
