using TopOpt, LinearAlgebra

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0 # filter radius

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
]
problem_names = ["Cantilever beam", "Half MBB beam", "L-beam", "Tie-beam"]

i = 1
println(problem_names[i])
problem = problems[i]

V = 0.5 # volume fraction
xmin = 0.001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
convcriteria = Nonconvex.KKTCriteria()
penalty = TopOpt.PowerPenalty(1.0)

solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

stress = TopOpt.von_mises_stress_function(solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin=rmin)
end
volfrac = TopOpt.Volume(solver)

x0 = ones(length(solver.vars))
threshold = 3 * maximum(stress(filter(PseudoDensities(x0))))

obj = x -> volfrac(filter(PseudoDensities(x)))
constr = x -> norm(stress(filter(PseudoDensities(x))), 5) - threshold

N = length(solver.vars)
x0 = fill(0.5, N)

options = MMAOptions(; maxiter=2000, tol=Nonconvex.Tolerance(; kkt=1e-4), convcriteria)
model = Model(obj)
addvar!(model, zeros(N), ones(N))
add_ineq_constraint!(model, constr)
alg = MMA87()
r = optimize(model, alg, x0; options)

@show obj(r.minimizer)
@show constr(r.minimizer)

using Makie
using CairoMakie

fig = visualize(problem; topology=r.minimizer)
Makie.display(fig)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
