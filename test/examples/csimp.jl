using TopOpt, Test

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[#PointLoadCantilever(Val{:Linear}, (60, 20, 20), (1.0, 1.0, 1.0), E, v, f), 
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64; force=f),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names = [
    #"3d cantilever beam",
    "cantilever beam",
    "half MBB beam",
    "L-beam",
    "tie-beam",
]

# NOTE: non-convexity + computational error lead to different solutions that satisfy the KKT tolerance
println("Continuation SIMP")
println("-"^10)

i = 1
println(problem_names[i])
# Define the problem
problem = problems[i]
# Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
rmin = 3.0
steps = 9 # number of penalty steps
convcriteria = Nonconvex.KKTCriteria()

x0 = fill(V, TopOpt.getncells(problem))
penalty = TopOpt.PowerPenalty(1.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
global comp = Compliance(solver)
global filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin=rmin)
end
global obj = x -> comp(filter(PseudoDensities(x)))
# Define volume constraint
global volfrac = Volume(solver)
global constr = x -> volfrac(filter(PseudoDensities(x))) - V
model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr)
alg = MMA87()

nsteps = 8
ps = range(1.0, 5.0; length=nsteps + 1)
tols = exp10.(range(-1, -3; length=nsteps + 1))
global x = x0
for j in 1:(nsteps+1)
    p = ps[j]
    tol = tols[j]
    TopOpt.setpenalty!(solver, p)
    options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=1000, convcriteria)
    res = optimize(model, alg, x; options)
    global x = res.minimizer
end
