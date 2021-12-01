using TopOpt
import Makie
using TopOpt.TopOptProblems.Visualization: visualize

using TimerOutputs

# function run_topopt()
    println("Start running.")
    # https://github.com/KristofferC/TimerOutputs.jl
    to = TimerOutput()
    reset_timer!(to)
    Nonconvex.show_residuals[] = true

    # Define the problem
    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 0.5; # downward force

    # Parameter settings
    V = 0.5 # volume fraction
    # xmin = 0.001 # minimum density
    xmin = 1e-6 # minimum density
    rmin = 0.04; # density filter radius

    nels =  (720, 240) # (720, 240) # (360, 120) # | (720, 240) 
    sizes = (3.0/nels[1], 1.0/nels[2])
    @timeit to "problem def" problem = HalfMBB(Val{:Linear}, nels, sizes, E, v, f);

    # Define a finite element solver
    @timeit to "penalty def" penalty = TopOpt.PowerPenalty(3.0)
    @timeit to "solver def" solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
        penalty = penalty);

    # Define compliance objective
    @timeit to "objective def" begin
        # Define compliance objective
        comp = Compliance(problem, solver)
        filter = DensityFilter(solver, rmin = rmin)
        obj = x -> comp(filter(x))
    end

    # Define volume constraint
    @timeit to "constraint def" begin
        volfrac = TopOpt.Volume(problem, solver)
        constr = x -> volfrac(filter(x)) - V
    end

    # Define subproblem optimizer
    # ! seems to be absolute diff
    mma_options = options = Nonconvex.MMAOptions(maxiter = 1000, 
        tol = Nonconvex.Tolerance(x = 1e-3, fabs = 1e-3, frel = 0.0, kkt = 1e-3),
        )
    convcriteria = Nonconvex.GenericCriteria()
    # convcriteria = Nonconvex.KKTCriteria()

    x0 = fill(V, length(solver.vars))
    @timeit to "optimizer def" optimizer = Optimizer(obj, constr, x0, Nonconvex.MMA87(),
        options = mma_options,
        convcriteria = convcriteria);

    # Define SIMP optimizer
    @timeit to "simp def" simp = SIMP(optimizer, solver, penalty.p);

    # Solve
    # initial solution, critical to set it to volfrac! (blame non-convexity :)
    @timeit to "simp run" result = simp(x0);

    # Print the timings in the default way
    println()
    show(to)

    @show result.convstate
    @show result.objval
    try
        @show optimizer.workspace.iter
    catch
        # IpoptWorkspace has no field iter
    end

    # # Visualize the result using Makie.jl
    fig = visualize(problem; topology=result.topology, 
        default_exagg_scale=0.07, scale_range=10.0, vector_linewidth=3, vector_arrowsize=0.005, 
        default_support_scale=0.01, default_load_scale=0.01)
    Makie.display(fig)

    # return problem, result
# end

# problem, result = run_topopt();