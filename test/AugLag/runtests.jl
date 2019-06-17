using Revise, FDM, Test
using TopOpt
using TopOpt.AugLag
using Test, LinearAlgebra

using TopOpt.AugLag: Constraint, Objective, LagrangianAlgorithm, AugmentedLagrangianAlgorithm, LagrangianFunction, AugmentedPenalty, IneqConstraintBlock, EqConstraintBlock
using TopOpt.Algorithms: BoxOptimizer

struct F <: Function
    grad
end
struct G <: Function
    grad
    a
    b
end
struct V <: Function
    grad
end
struct BoxOpt
    obj
    lb
    ub
end
function (o::BoxOpt)(x)
    obj, lb, ub = o.obj, o.lb, o.ub
    func = x -> obj(x, similar(x))
    grad_func = (g, x) -> (obj(x, g); g)
    Optim.optimize(func, grad_func, lb, ub, x, Optim.Fminbox(ConjugateGradient()), Optim.Options(x_tol=-0.1, f_tol=-0.1, allow_f_increases=true))
end
function TopOpt.Algorithms.setbounds!(o::BoxOpt, x, w)
    o.lb .= max.(0, x .- w)
    o.ub .= min.(1, x .+ w)
end

struct MMAOpt{TM}
    model::TM
end
function (o::MMAOpt)(x)
    model = o.model
    options = MMA.Options(tol = MMA.Tolerances(xtol = -1e-6, ftol = -1e-6, grtol = -1e-6, kkttol = 1e-6))
    convcriteria = MMA.KKTCriteria()
    workspace = MMA.Workspace(model, x, MMA87(), ConjugateGradient(); options = options, convcriteria = convcriteria)
    MMA.optimize!(workspace)
end
function TopOpt.Algorithms.setbounds!(o::MMAOpt, x, w)
    o.model.box_min .= max.(0, x .- w)
    o.model.box_max .= min.(1, x .+ w)
end

function (f::F)(x::AbstractVector, grad::AbstractVector=f.grad)
    if length(grad) != 0
        grad[1] = 0.0
        grad[2] = 0.5/sqrt(x[2])
    end
    sqrt(x[2])
end

function (g::G)(x::AbstractVector, grad::AbstractVector=g.grad)
    if length(grad) != 0
        grad[1] = 3g.a * (g.a*x[1] + g.b)^2
        grad[2] = -1
    end
    (g.a*x[1] + g.b)^3 - x[2]
end

function (v::V)(x::AbstractVector, grad::AbstractVector=v.grad)
    if length(grad) != 0
        grad .= 1
    end
    sum(x)
end

ndim = 2

@testset "Lagrangian function gradient" begin
    f = F(zeros(ndim))
    g = G(zeros(ndim), 2, 0)

    obj = Objective(f)
    constr = Constraint(g, 0.0)

    ineq_block = IneqConstraintBlock((constr,), [1000.0], [0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, Ref(0.1))

    x = rand(2)*10
    grad = similar(x)

    lag = Objective(LagrangianFunction(obj, pen))
    lag(x, grad)
    grad_fdm = FDM.grad(central_fdm(5, 1), x -> lag(x, similar(x)), x)
    @test isapprox(grad_fdm, grad, rtol=1e-5)

    linpen = AugLag.LinearPenalty(eq_block, ineq_block)
    linpen(x, grad)
    grad_fdm = FDM.grad(central_fdm(5, 1), x -> linpen(x, similar(x)), x)
    @test isapprox(grad_fdm, grad, rtol=1e-5)

    grad_g = similar(grad)
    grad_constr = similar(grad)
    g1(x, grad_g)
    constr1(x, grad_constr)
    @test isapprox(grad_constr, grad_g, rtol = 1e-5)

    ineq_block = IneqConstraintBlock((constr,), [0.0], [0.0])
    augpen = AugLag.AugmentedPenalty(eq_block, ineq_block, Ref(10.0))
    pval = augpen(x, grad)
    grad_fdm = FDM.grad(central_fdm(5, 1), x -> augpen(x, similar(x)), x)
    @test pval == 10*max(constr1(x, similar(x)), 0)^2
    @test isapprox(grad_fdm, grad, rtol=1e-5)
    @test isapprox(2*10*sqrt(pval/10)*grad_g, grad_fdm, rtol=1e-5)
end

@testset "Lagrangian algorithm - Box" begin
    f = F(zeros(ndim))
    g1 = G(zeros(ndim), 2, 0)
    g2 = G(zeros(ndim), -1, 1)

    obj = Objective(f)
    constr1 = Constraint(g1, 0.0)
    constr2 = Constraint(g2, 0.0)

    ineq_block = IneqConstraintBlock((constr1, constr2), [0.0, 0.0], [0.0, 0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, Ref(1.0))
    lag = LagrangianFunction(obj, pen)

    x = [0.9, 0.9]
    grad = similar(x)

    lb = zeros(2); ub = ones(2);
    optimizer = BoxOpt(lag, lb, ub)
    
    w = 0.25; gamma=1.5; alpha=0.1
    alg = AugmentedLagrangianAlgorithm(optimizer, lag, 20, copy(x), w, gamma, alpha)
    AugLag.reset!(alg);
    result = alg(x)
    @test constr1(result.minimizer) < 1e-3
    @test constr2(result.minimizer) < 1e-3
    @test abs(obj(result.minimizer) - sqrt(8/27)) < 1e-3
    @test norm(result.minimizer - [1/3, 8/27]) < 1e-3
end

@testset "Lagrangian algorithm - Box - Compliance" begin
    problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(Displacement, Direct, problem, xmin = 0.001, penalty = TopOpt.PowerPenalty(1.0))

    comp = ComplianceFunction(problem, solver, filtering = true, rmin = 4.0,
        tracing = true, logarithm = false)
    constr = Constraint(VolumeFunction(problem, solver), 0.5)
    ineq_block = IneqConstraintBlock((constr,), [0.0], [0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, Ref(1.0))

    obj = LagrangianFunction(comp, pen)
    x = similar(solver.vars); x .= 0.9;
    optimizer = BoxOptimizer(obj, GradientDescent())
    #=
    mma_options = options = MMA.Options(maxiter = 1000, 
        tol = MMA.Tolerances(kkttol = 0.001))
    convcriteria = MMA.KKTCriteria()
    optimizer = MMAOptimizer(Objective(obj), constr, MMA.MMA87(),
        ConjugateGradient(), options = mma_options,
        convcriteria = convcriteria)
    =#

    w = 0.5; gamma=2.0; alpha=1.0
    alg = AugmentedLagrangianAlgorithm(optimizer, obj, 20, copy(x), w, gamma, alpha)
    AugLag.reset!(alg);
    alg.lag.penalty.r[] = 1.0
    result = alg(x, verbose=true)

    for p in 1.2:0.2:3.0
        global optimizer, x, result, alg, obj, w, gamma, alpha
        TopOpt.setpenalty!(optimizer, p)
        x = copy(result.minimizer)
        alg = AugmentedLagrangianAlgorithm(optimizer, obj, 20, copy(x), w, gamma, alpha)
        AugLag.reset!(alg);
        alg.lag.penalty.r[] = 1.0
        result = alg(x, verbose=true)
    end
    @test constr(result.minimizer) < 1e-3
end

# Currently gives a sub-optimal feasible solution
# First order KKT conditions are satisfied
#=
@testset "Lagrangian algorithm - MMA" begin
    f = F(zeros(ndim))
    g1 = G(zeros(ndim), 2, 0)
    g2 = G(zeros(ndim), -1, 1)
    v = V(zeros(ndim))

    obj = Objective(f)
    constr1 = Constraint(g1, 0.0)
    constr2 = Constraint(g2, 0.0)
    constr3 = Constraint(v, 0.65)

    x = [0.9, 0.9]
    
    ineq_block = IneqConstraintBlock((constr1, constr2), [0.0, 0.0], [0.0, 0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, Ref(1.0))
    lag = LagrangianFunction(obj, pen)

    model = MMA.Model(ndim, lag)
    box!(model, 1, 0.0, 1.0)
    box!(model, 2, 0.0, 1.0)    
    ineq_constraint!(model, constr3)
    optimizer = MMAOpt(model)

    w = 0.2; gamma=1.3; alpha=0.5
    alg = AugmentedLagrangianAlgorithm(optimizer, lag, 20, copy(x), w, gamma, alpha)
    AugLag.reset!(alg);
    result = alg(x)
    
    @test constr1(result.minimizer) < 1e-3
    @test constr2(result.minimizer) < 1e-3
    @test abs(obj(result.minimizer) - sqrt(8/27)) < 1e-3
    @test norm(result.minimizer - [1/3, 8/27]) < 1e-3    
end
=#
