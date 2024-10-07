module ContComplianceDemo1

using TopOpt, LinearAlgebra, StatsFuns
using Makie
using CairoMakie
# using GLMakie

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
problem_size = (60, 20)
V = 0.5 # maximum volume fraction
p = 4.0 # penalty
x0 = fill(V, prod(problem_size)) # initial design

problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=xmin)
cheqfilter = DensityFilter(solver; rmin=rmin)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize compliance
    return comp(cheqfilter(PseudoDensities(x)))
end
function constr(x)
    # volume fraction constraint
    return sum(cheqfilter(PseudoDensities(x))) / length(x) - V
end

m = Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, x=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
# Method of Moving Asymptotes
@time r = Nonconvex.optimize(m, MMA87(), x0; options=options)

@show obj(r.minimizer)
@show constr(r.minimizer)
topology = cheqfilter(PseudoDensities(r.minimizer)).x
fig = visualize(
    problem; solver.u, topology=topology, default_exagg_scale=0.0, scale_range=10.0
)
Makie.display(fig)

end
