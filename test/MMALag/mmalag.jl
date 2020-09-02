using Revise, TopOpt2, Distributions, LinearAlgebra, Random, PyPlot, FileIO, JLD2
using TopOpt2.AugLag: AugLag, IneqConstraintBlock, EqConstraintBlock, AugmentedPenalty, Lagrangian, AugmentedLagrangianAlgorithm, LinQuadAggregation
using TopOpt2.Algorithms: BoxOptimizer

# Setup
    Random.seed!(1)
    E = 1.0; v = 0.3; xmin = 0.001;
    filterT = DensityFilter
    rmin = 2.0; V = 0.3

    f1 = RandomMagnitude([0, -1], Uniform(0.5, 1.5))
    f2 = RandomMagnitude(normalize([1, -1]), Uniform(0.5, 1.5))
    f3 = RandomMagnitude(normalize([-1, -1]), Uniform(0.5, 1.5))

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    kwargs = (filterT = filterT, tracing = true, logarithm = false, rmin = rmin)
    base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v)
    problem = MultiLoad(base_problem, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3], 1000)
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)

# MMA-Lag

func = identity
multiple1 = 1.0
exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
x = ones(length(solver.vars))
exact_svd_bc = exact_svd_block(x)
multiple2 = 1/maximum(abs, exact_svd_bc)

ps = 1.0:0.5:6.0
maxtol = 1e-3
mintol = 1e-4
steps = length(ps)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 1:length(ps)
    p = ps[i]
    h = hs[1]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end

hs = 0.0:4:20.0
steps = length(hs)
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
ftol_gen = ExponentialContinuation(a, b, 0.0, steps+1, mintol)

for i in 2:length(hs)
    p = ps[end]
    h = hs[i]
    tol = ftol_gen(i)
    println("=======")
    @show p, h, tol
    println("=======")
    global x, projection
    projection.β = h
    penalty = ProjectedPenalty(PowerPenalty(p), projection);
    TopOpt.setpenalty!(solver, penalty)
    global result = alg(outer_iterations = 10, inner_iterations = 50, primal_alpha0 = primal_alpha0, dual_alpha0 = dual_alpha0, ftol=tol, xtol=1e-4, trust_region = w, primal_optim = primal_optim, gamma=gamma, adapt_primal_step=adapt_primal_step, adapt_dual_step=adapt_dual_step, primal_cg=primal_cg, primal_step_adapt_factor0=primal_step_adapt_factor0, dual_step_adapt_factor0=dual_step_adapt_factor0, adapt_trust_region=adapt_trust_region)
    x = copy(result.minimizer)
    AugLag.reset!(alg, r = 0.1, x = x);
    if TopOpt.TopOptProblems.getdim(problem) == 2
        image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(x)))
        if i == 1
            global _im = PyPlot.imshow(image, cmap="Greys", origin="lower")
        else
            _im.set_data(image)
        end
    end
end



fname = "$out_dir/exact_svd_max_csimp"
if TopOpt.TopOptProblems.getdim(problem) == 2
    image = TopOptProblems.RectilinearTopology(problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))
    PyPlot.imshow(image, cmap="Greys", origin="lower")
    savefig("$fname.png")
    close()
end
save(
    "$fname.jld2",
    Dict("problem" => problem, "result" => result),
)
println("...............")
save_mesh(fname, problem, solver.penalty.proj.(filterT(Val(true), solver, rmin)(result.topology)))

@show maximum(exact_svd_block(result.minimizer))
@show obj(result.minimizer)

func = identity
    multiple1 = 1.0
    exact_svd_block = BlockCompliance(problem, solver; kwargs..., method = :exact_svd)
    x = ones(length(solver.vars))
    exact_svd_bc = exact_svd_block(x)
    #multiple2 = 1/maximum(abs, exact_svd_bc)
    multiple2 = 1.0

    projection = HeavisideProjection(0.0)
    penalty = ProjectedPenalty(PowerPenalty(1.0), projection);
    obj = Objective(Volume(problem, solver, filterT = DensityFilter, rmin = rmin, postproj = projection))
    constr = BlockConstraint(multiple2 * func(exact_svd_block), multiple2 * func(5000.0))
    mma_options = MMA.Options(maxiter = 500, tol = MMA.Tolerances(kkttol = 1e-4), s_init = 0.1, s_incr = 1.1, s_decr = 0.9)
    convcriteria = MMA.IpoptCriteria()
    optimizer = MMAOptimizer(obj, constr, MMALag.MMALag20(MMA.MMA87(), true), ConjugateGradient(), options = mma_options, convcriteria = convcriteria)

    x0 = fill(1.0, 160*40);
    ps = 1.0:1.0:6.0
    hs = 0.0:4.0:20.0
    i = 1
    for i in 1:length(ps)
        p = ps[i]
        h = hs[i]
        projection.β = h
        penalty = ProjectedPenalty(PowerPenalty(p), projection);
        TopOpt.setpenalty!(solver, penalty)
        simp = SIMP(optimizer, penalty.p)
        if i == 1
            global result = simp(x0)
        else
            global result = simp(result.topology)
        end
    end
