using TopOpt, LinearAlgebra, StatsFuns
using Makie
using CairoMakie
# using GLMakie

Nonconvex.@load Juniper

# 2D
ndim = 2
node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
    joinpath(@__DIR__, "tim_$(ndim)d.json")
);
ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
loads = load_cases["0"]
problem = TrussProblem(
    Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
);

xmin = 0.0001 # minimum density
x0 = fill(1.0, ncells) # initial design
p = 1.0 # penalty
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
addvar!(m, zeros(length(x0)), ones(length(x0)); integer=trues(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = JuniperIpoptOptions()
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(m, JuniperIpoptAlg(), x0; options=options);

@show obj(r.minimizer)
@show constr(r.minimizer)
fig = visualize(problem; solver.u, topology=r.minimizer, default_exagg_scale=0.0)
Makie.display(fig)
