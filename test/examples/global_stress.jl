# using Revise
using TopOpt, LinearAlgebra

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
problem_names = [
    "Cantilever beam",
    "Half MBB beam",
    "L-beam",
    "Tie-beam",
]

println("Global Stress")
println("-"^10)

for i in 1:length(problems)
    println(problem_names[i])
    problem = problems[i]
    # Parameter settings
    V = 0.5 # volume fraction
    xmin = 0.001 # minimum density
    steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
    convcriteria = Nonconvex.KKTCriteria()
    penalty = TopOpt.PowerPenalty(1.0)
    # Define a finite element solver
    solver = FEASolver(
        Displacement, Direct, problem, xmin = xmin, penalty = penalty,
    )
    # Define compliance objective
    stress = TopOpt.MicroVonMisesStress(solver)
    filter = if problem isa TopOptProblems.TieBeam
        identity
    else
        DensityFilter(solver, rmin = rmin)
    end
    volfrac = Volume(problem, solver)

    obj = x -> volfrac(filter(x))
    constr = x -> norm(stress(filter(x)), 5) - 1.0
    # Define subproblem optimizer
    x0 = fill(1.0, length(solver.vars))
    options = Nonconvex.MMAOptions(
        maxiter=2000, tol = Nonconvex.Tolerance(kkt = 1e-4),
    )
    #options = Nonconvex.PercivalOptions()
    optimizer = Optimizer(
        obj, constr, x0, Nonconvex.MMA87(), #Nonconvex.PercivalAlg(),
        options = options, convcriteria = convcriteria,
    )

    # Define continuation SIMP optimizer
    simp = SIMP(optimizer, solver, 3.0)
    # Solve
    result = simp(x0)
end
