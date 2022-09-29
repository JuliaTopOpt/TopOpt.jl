# using Revise
using TopOpt, LinearAlgebra, Nonconvex
Nonconvex.@load NLopt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names = ["Cantilever beam", "Half MBB beam", "L-beam", "Tie-beam"]

println("Global Stress")
println("-"^10)

i = 1
# for i in 1:length(problems)
println(problem_names[i])
problem = problems[i]
# Parameter settings
xmin = 0.001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
penalty = TopOpt.PowerPenalty(3.0)
# Define a finite element solver
solver = FEASolver(Direct, problem; xmin = xmin, penalty = penalty)
stress = TopOpt.von_mises_stress_function(solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin = rmin)
end
volfrac = Volume(solver)

obj = x -> volfrac(filter(PseudoDensities(x)))
nvars = length(solver.vars)
x0 = fill(1.0, nvars)
threshold = 3 * maximum(stress(filter(PseudoDensities(x0))))
constr = x -> begin
    norm(stress(filter(PseudoDensities(x))), 10) - threshold
end

alg = NLoptAlg(:LD_MMA)
options = NLoptOptions()
model = Nonconvex.Model(obj)
Nonconvex.addvar!(model, zeros(nvars), ones(nvars))
Nonconvex.add_ineq_constraint!(model, constr)
r = Nonconvex.optimize(model, alg, x0; options)
@show obj(r.minimizer)
# end
