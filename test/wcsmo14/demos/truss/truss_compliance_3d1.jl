module TrussComplianceDemo3D1

using TopOpt, LinearAlgebra, StatsFuns
# using Makie, GLMakie
# using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

# 3D
ndim = 3
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_$(ndim)d.json")
)
ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
)

xmin = 0.0001 # minimum density
x0 = fill(1.0, ncells) # initial design
p = 4.0 # penalty
V = 0.5 # maximum volume fraction

solver = FEASolver(Direct, problem; xmin=xmin)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize compliance
    return comp(PseudoDensities(x))
end
function constr(x)
    # volume fraction constraint
    return sum(x) / length(x) - V
end

m = Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)

@show r.minimizer
@show obj(r.minimizer)
@show constr(r.minimizer)
# fig = visualize(
#     problem; u=solver.u, topology = r.minimizer,
#     default_exagg_scale=0.0, exagg_range=100.0,
# )
# Makie.display(fig)

end
