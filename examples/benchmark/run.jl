using TopOpt
using TopOpt.TopOptProblems.Visualization: visualize
include("./new_problems.jl")

using TimerOutputs
import Makie

function run_topopt()
    # https://github.com/KristofferC/TimerOutputs.jl
    to = TimerOutput()
    reset_timer!(to)

    # ### Define the problem
    E = 1.0 # Young’s modulus
    v = 0.3 # Poisson’s ratio
    f = 1.0; # downward force

    # nels = (30, 10, 4) 
    # sizes = (1.0, 1.0, 1.0)
    nels = (160, 40) 
    sizes = (1.0, 1.0)
    # @timeit to "problem def" problem = NewTopOptProblems.NewPointLoadCantilever(Val{:Linear}, nels, sizes, E, v, f);
    @timeit to "problem def" problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, v, f);

    # ### Parameter settings
    V = 0.3 # volume fraction
    xmin = 0.001 # minimum density
    rmin = 4.0; # density filter radius

    # ### Define a finite element solver
    @timeit to "penalty def" penalty = TopOpt.PowerPenalty(3.0)
    @timeit to "solver def" solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
        penalty = penalty);

    # ### Define compliance objective
    @timeit to "objective def" obj = Objective(TopOpt.Compliance(problem, solver, filterT = DensityFilter,
        rmin = rmin, tracing = true, logarithm = false));

    # ### Define volume constraint
    @timeit to "constraint def" constr = Constraint(TopOpt.Volume(problem, solver, filterT = DensityFilter, rmin = rmin), V);

    # ### Define subproblem optimizer
    # mma_options = options = MMA.Options(maxiter = 200, 
    #     tol = MMA.Tolerances(kkttol = 0.01))
    mma_options = options = MMA.Options(maxiter = 3000, 
        tol = MMA.Tolerances(kkttol = 0.001))
    convcriteria = MMA.KKTCriteria()
    @timeit to "optimizer def" optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
        ConjugateGradient(), options = mma_options,
        convcriteria = convcriteria);

    # ### Define SIMP optimizer
    @timeit to "simp def" simp = SIMP(optimizer, penalty.p);

    # ### Solve
    x0 = fill(1.0, length(solver.vars))
    @timeit to "simp run" result = simp(x0);

    # Print the timings in the default way
    show(to)

    @show result.convstate

    # ### Visualize the result using Makie.jl
    fig = visualize(problem; topology=result.topology, 
        default_exagg_scale=0.07, scale_range=10.0, vector_linewidth=3, vector_arrowsize=0.5)
    Makie.display(fig)
end

run_topopt()