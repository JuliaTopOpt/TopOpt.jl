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
volfrac = TopOpt.Volume(problem, solver)

x0 = ones(length(solver.vars))
threshold = 2 * maximum(stress(filter(x0)))

obj = x -> volfrac(filter(x))
constr = x -> norm(stress(filter(x)), 5) - threshold
options = MMAOptions(; maxiter=2000, tol=Nonconvex.Tolerance(; kkt=1e-4))

x0 = fill(0.5, length(solver.vars))
optimizer = Optimizer(obj, constr, x0, MMA87(); options=options, convcriteria=convcriteria)

simp = SIMP(optimizer, solver, 3.0)

result = simp(x0)

@show result.convstate
@show optimizer.workspace.iter
@show result.objval

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

