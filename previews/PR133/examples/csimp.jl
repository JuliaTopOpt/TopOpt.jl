using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20, 20), (1.0, 1.0, 1.0), E, v, f),
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64; force = f),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names =
    ["3d cantilever beam", "cantilever beam", "half MBB beam", "L-beam", "tie-beam"]

i = 2
println(problem_names[i])
problem = problems[i]

V = 0.5 # volume fraction
xmin = 0.001 # minimum density
rmin = 3.0

convcriteria = Nonconvex.KKTCriteria()
x0 = fill(V, TopOpt.getncells(problem))
penalty = TopOpt.PowerPenalty(1.0)
solver = FEASolver(Direct, problem; xmin = xmin, penalty = penalty)
comp = Compliance(solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin = rmin)
end
obj = x -> comp(filter(PseudoDensities(x)))

volfrac = Volume(solver)
constr = x -> volfrac(filter(PseudoDensities(x))) - V
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()

nsteps = 4
ps = range(1.0, 5.0; length = nsteps + 1)

tols = exp10.(range(-2, -4; length = nsteps + 1))
x = x0
for j = 1:(nsteps+1)
    global convcriteria
    p = ps[j]
    tol = tols[j]
    TopOpt.setpenalty!(solver, p)
    options = MMAOptions(; tol = Tolerance(; kkt = tol), maxiter = 1000, convcriteria)
    res = optimize(model, alg, x; options)
    global x = res.minimizer
end

@show obj(x)
@show constr(x)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

