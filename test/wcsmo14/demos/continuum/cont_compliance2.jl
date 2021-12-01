module ContComplianceDemo2

using Makie, TopOpt, LinearAlgebra, StatsFuns
using TopOpt.TopOptProblems.Visualization: visualize

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 4.0 # filter radius
xmin = 0.0001 # minimum density
problem_size = (160, 40)
x0 = fill(1.0, prod(problem_size)) # initial design
p = 4.0 # penalty
compliance_threshold = 800 # maximum compliance

problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)
#problem = HalfMBB(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f)

solver = FEASolver(Displacement, Direct, problem, xmin = xmin)

cheqfilter = DensityFilter(solver, rmin = rmin)
stress = TopOpt.MicroVonMisesStress(solver)
comp = TopOpt.Compliance(problem, solver)

function obj(x)
    # minimize volume
    return sum(cheqfilter(x)) / length(x)
end
function constr(x)
    # compliance upper-bound
    return comp(cheqfilter(x)) - compliance_threshold
end

m = Model(obj)
addvar!(m, zeros(length(x0)), ones(length(x0)))
Nonconvex.add_ineq_constraint!(m, constr)

options = MMAOptions(
    maxiter=1000, tol = Tolerance(kkt = 1e-4, x=1e-4, f = 1e-4),
)
TopOpt.setpenalty!(solver, p)
@time r = Nonconvex.optimize(
    m, MMA87(dualoptimizer = ConjugateGradient()),
    x0, options = options,
);

@show obj(r.minimizer)
@show constr(r.minimizer)
@show maximum(stress(cheqfilter(r.minimizer)))
topology = cheqfilter(r.minimizer);
fig = visualize(problem, solver.u; 
    topology = topology, default_exagg_scale=0.0, scale_range=10.0)
Makie.display(fig)

end
