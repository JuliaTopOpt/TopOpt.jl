E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f), 
            HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f), 
            LBeam(Val{:Linear}, Float64), 
            TieBeam(Val{:Quadratic}, Float64)]
problem_names = ["cantilever beam", "half MBB beam", "L-beam", "tie-beam"]
approx_objvals = [187.0, 87.0, 55.0, 847.0]

#@testset "Continuation SIMP - $(problem_names[i])" for i in 1:4
    i = 1
    # Define the problem
    problem = problems[i]
    # Parameter settings
    V = 0.5 # volume fraction
    xmin = 0.001 # minimum density
    maxtol = 0.01 # maximum tolerance
    mintol = 0.0001 # minimum tolerance
    steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
    reuse = true # adaptive penalty flag

    # Define a finite element solver
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
    penalty = TopOpt.PowerPenalty(1.0))
    # Define compliance objective
    filtering = problem isa TopOptProblems.TieBeam ? false : true
    obj = ComplianceObj(problem, solver, filtering = filtering,
    rmin = 3.0, tracing = true, logarithm = false)
    # Define volume constraint
    constr = VolConstr(problem, solver, V)
    # Define subproblem optimizer
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
        ConjugateGradient(), maxiter=1000); optimizer.obj.fevals = 0

    # Define continuation SIMP optimizer
    simp = SIMP(optimizer, 1.0)
    b = log(mintol / maxtol) / steps
    a = maxtol / exp(b)
    ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)
    cont_simp = ContinuationSIMP(simp, start=1.0, steps=steps,
    finish=5.0, reuse=reuse, ftol_cont=ftol_gen)
    # Solve
    x0 = fill(1.0, length(solver.vars))
    result = cont_simp(x0)

    @test round(result.objval, 0) == approx_objvals[i]
#end
