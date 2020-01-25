E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[PointLoadCantilever(Val{:Linear}, (80, 20, 20), (1.0, 1.0, 1.0), E, v, f), 
            PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
            HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f), 
            LBeam(Val{:Linear}, Float64), 
            TieBeam(Val{:Quadratic}, Float64)]
problem_names = ["cantilever beam", "half MBB beam", "L-beam", "tie-beam"]
approx_objvals = [330.0, 175.0, 65.0, 1413.0]

@testset "Continuation SIMP - $(problem_names[i])" for i in 1:4
    # Define the problem
    problem = problems[i]
    # Parameter settings
    V = 0.5 # volume fraction
    xmin = 0.001 # minimum density
    steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
    reuse = true # adaptive penalty flag
    convcriteria = MMA.KKTCriteria()
    #penalty = TopOpt.PowerPenalty(1.0)
    penalty = TopOpt.PowerPenalty(1.0)
    pcont = Continuation(penalty, steps = steps, xmin = xmin, pmax = 5.0)

    mma_options = options = MMA.Options(maxiter=1000)
    if convcriteria isa MMA.KKTCriteria
        maxtol = 0.1 # maximum tolerance
        mintol = 0.001 # minimum tolerance

        b = log(mintol / maxtol) / steps
        a = maxtol / exp(b)
        mma_options_gen = TopOpt.MMAOptionsGen(steps = steps, initial_options = mma_options, kkttol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol))
    else
        maxtol = 0.01 # maximum tolerance
        mintol = 0.0001 # minimum tolerance

        b = log(mintol / maxtol) / steps
        a = maxtol / exp(b)
        mma_options_gen = TopOpt.MMAOptionsGen(steps = steps, initial_options = mma_options, ftol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol))
    end
    csimp_options = TopOpt.CSIMPOptions(steps = steps, 
                                        options_gen = mma_options_gen, 
                                        p_gen = pcont, 
                                        reuse = reuse
                                        )

    # Define a finite element solver
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)
    # Define compliance objective
    filtering = problem isa TopOptProblems.TieBeam ? false : true
    obj = Objective(Compliance(problem, solver, filtering = filtering,
        rmin = 3.0, tracing = true, logarithm = false))
    #cu_obj = TopOpt.cu(obj)
    # Define volume constraint
    constr = Constraint(Volume(problem, solver), V)
    # Define subproblem optimizer
    #optimizer = MMAOptimizer{CPU}(cu_obj, constr, MMA.MMA87(),
    #    ConjugateGradient(), maxiter=1000); optimizer.obj.fevals = 0
    optimizer = MMAOptimizer{CPU}(obj, constr, MMA.MMA87(),
        ConjugateGradient(), options = mma_options, convcriteria = convcriteria); optimizer.obj.f.fevals = 0

    # Define continuation SIMP optimizer
    simp = SIMP(optimizer, penalty.p)

    cont_simp = ContinuationSIMP(simp, steps, csimp_options) 

    # Solve
    x0 = fill(1.0, length(solver.vars))
    result = cont_simp(x0)

    @test round(result.objval, digits=0) == approx_objvals[i]
end

@testset "Continuation SIMP 2 - $(problem_names[i])" for i in 1:4
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
    solver = FEASolver(Displacement, CG, MatrixFree, problem, xmin = xmin,
        penalty = TopOpt.PowerPenalty(1.0))
    # Define volume constraint
    obj = Objective(Volume(problem, solver))
    # Define compliance objective
    filtering = problem isa TopOptProblems.TieBeam ? false : true
    constr = Constraint(Compliance(problem, solver, filtering = filtering,
        rmin = 3.0, tracing = true, logarithm = false), approx_objvals[i])
    cu_constr = TopOpt.cu(constr)
    # Define subproblem optimizer

    mma_options = options = MMA.Options(maxiter=1000)
    optimizer = MMAOptimizer{CPU}(obj, constr, MMA.MMA87(),
        ConjugateGradient(), options = mma_options); optimizer.obj.f.fevals = 0

    # Define continuation SIMP optimizer
    simp = SIMP(optimizer, 1.0)

    b = log(mintol / maxtol) / steps
    a = maxtol / exp(b)
    mma_options_gen = TopOpt.MMAOptionsGen(steps = steps, initial_options = mma_options, ftol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol))
    csimp_options = TopOpt.CSIMPOptions(steps = steps, options_gen = mma_options_gen, pstart = 1.0, pfinish = 5.0)

    cont_simp = ContinuationSIMP(simp, steps, csimp_options) 

    # Solve
    x0 = fill(1.0, length(solver.vars))
    result = cont_simp(x0)

    @test round(result.objval, digits=0) == approx_objvals[i]
end
