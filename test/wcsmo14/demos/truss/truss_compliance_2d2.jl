# module TrussComplianceDemo2D2

using TopOpt, LinearAlgebra, StatsFuns
#using Makie, GLMakie
#using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

# 2D
ndim = 2
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
compliance_threshold = 5.0 # maximum compliance

solver = FEASolver(Direct, problem; xmin=xmin)
comp = TopOpt.Compliance(solver)

function obj(x)
    # minimize volume
    return sum(x) / length(x)
end
function constr(x)
    # compliance upper-bound
    return comp(PseudoDensities(x)) - compliance_threshold
end

m = Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(; maxiter=1000, tol=Tolerance(; kkt=1e-4, f=1e-4))
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(; dualoptimizer=ConjugateGradient()), x0; options=options
)

@show obj(r.minimizer)
@show constr(r.minimizer)
#fig = visualize(
#    problem, solver.u, topology = r.minimizer,
#    default_exagg_scale=0.0
#)
#Makie.display(fig)

# end
