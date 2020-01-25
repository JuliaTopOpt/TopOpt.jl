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
struct BoxOpt
    obj
    lb
    ub
end
function (o::BoxOpt)(x)
    obj, lb, ub = o.obj, o.lb, o.ub
    func = x -> obj(x, similar(x))
    grad_func = (g, x) -> (obj(x, g); g)
    Optim.optimize(func, grad_func, lb, ub, x, Optim.Fminbox(ConjugateGradient()), Optim.Options(x_tol=-0.1, f_tol=-0.1, allow_f_increases=false))
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
    options = MMA.Options(tol = MMA.Tolerances(kkttol = 1e-6))
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

ndim = 2

function test_grad(f, x)
    grad1 = similar(x)
    f(x, grad1)
    grad2 = FDM.grad(central_fdm(5, 1), x -> f(x, similar(x)), x)
    @test isapprox(grad1, grad2, rtol = 1e-5)
end

@testset "Lagrangian function gradient" begin
    f = F(zeros(ndim))
    g = G(zeros(ndim), 2, 0)

    obj = Objective(f)
    constr = Constraint(g, 0.0)

    ineq_block = IneqConstraintBlock((constr,), [1000.0], [0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, 0.1)

    x = rand(2)*10
    grad = similar(x)

    # Augmented Lagrangian objective
    lag = Objective(LagrangianFunction(obj, pen))
    test_grad(lag, x)

    # Linear penalty
    linpen = AugLag.LinearPenalty(eq_block, ineq_block)
    test_grad(linpen, x)

    # Aggregations
    agg1 = LinAggregation(lag, [0.5])
    agg2 = QuadAggregation(lag, 0.5, max=false)
    agg3 = QuadAggregation(lag, 0.5, max=true)
    agg4 = LinQuadAggregation(lag, [0.5], 0.5, max=false)
    agg5 = LinQuadAggregation(lag, [0.5], 0.5, max=true)
    test_grad(agg1, x)
    test_grad(agg2, x)
    test_grad(agg3, x)
    test_grad(agg4, x)
    test_grad(agg5, x)

    # Feasible
    agg3([0.1, 0.1])
    @test agg3.grad == [0.0, 0.0]

    # Dual with 0 objective
    x = rand(2)*10
    function lag_dual(λ, grad_λ)
        agg1.weights .= λ
        f = agg1(x)
        grad_λ .= agg1.fval; 
        return f
    end
    test_grad(lag_dual, [0.5])

    # Function vs constraint
    grad_g = similar(grad)
    grad_constr = similar(grad)
    g(x, grad_g)
    constr(x, grad_constr)
    @test isapprox(grad_constr, grad_g, rtol = 1e-5)

    # Augmented penalty function
    ineq_block = IneqConstraintBlock((constr,), [0.0], [0.0])
    augpen = AugLag.AugmentedPenalty(eq_block, ineq_block, 10.0)
    pval = augpen(x, grad)
    grad_fdm = FDM.grad(central_fdm(5, 1), x -> augpen(x, similar(x)), x)
    @test pval == 10*max(constr(x, similar(x)), 0)^2
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
    pen = AugmentedPenalty(eq_block, ineq_block, 1.0)
    lag = LagrangianFunction(obj, pen)

    x = [0.9, 0.9]
    grad = similar(x)

    lb = zeros(2); ub = ones(2);
    optimizer = BoxOpt(lag, lb, ub)
    
    w = 0.25; gamma=1.5; alpha=10.0
    alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x))
    AugLag.reset!(alg);
    result = alg(x, outer_iterations=20, trust_region=w, gamma=gamma, alpha=alpha)
    @test constr1(result.minimizer) < 1e-3
    @test constr2(result.minimizer) < 1e-3
    @test abs(result.minimum - sqrt(8/27)) < 1e-3
    @test norm(result.minimizer - [1/3, 8/27]) < 1e-3
end

@testset "Lagrangian algorithm - Box - Small Compliance" begin
    problem = PointLoadCantilever(Val{:Linear}, (2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(Displacement, Direct, problem, xmin = 0.001, penalty = TopOpt.PowerPenalty(1.0))

    x = similar(solver.vars); x .= 0.7;

    comp = Compliance(problem, solver, filtering = false,
        rmin = 4.0, tracing = true, logarithm = false)
    test_grad(comp, x)

    vol = Volume(problem, solver)
    test_grad(vol, x)

    bin_pen = BinPenalty(solver, 0.5)
    test_grad(bin_pen, x)

    constr = Constraint(Volume(problem, solver), 0.3)
    ineq_block = IneqConstraintBlock((constr,), [100.0], [0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, 100.0)
    obj = Lagrangian(comp, pen)

    optimizer = BoxOptimizer(obj)
    w = 1.0; gamma=2.0; alpha=0.5
    alg = AugmentedLagrangianAlgorithm(optimizer, obj, copy(x))
    AugLag.reset!(alg);
    alg.lag.penalty.r[] = 1.0
    result = alg(x, outer_iterations=20, trust_region=w, gamma=gamma, alpha=alpha)

    @test constr(result.minimizer) < 1e-3
end

@testset "Lagrangian algorithm - MMA" begin
    f = F(zeros(ndim))
    g1 = G(zeros(ndim), 2, 0)
    g2 = G(zeros(ndim), -1, 1)

    obj = Objective(f)
    constr1 = Constraint(g1, 0.0)
    constr2 = Constraint(g2, 0.0)

    x = [0.9, 0.9]
    
    ineq_block = IneqConstraintBlock((constr1,), [0.0], [0.0])
    eq_block = EqConstraintBlock((), [], [])
    pen = AugmentedPenalty(eq_block, ineq_block, 1.0)
    lag = LagrangianFunction(obj, pen)

    model = MMA.Model(ndim, lag)
    box!(model, 1, 0.0, 1.0)
    box!(model, 2, 0.0, 1.0)    
    ineq_constraint!(model, constr2)
    optimizer = MMAOpt(model)

    w = 1.0; gamma=1.2; alpha=1.0
    alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x))
    AugLag.reset!(alg);
    result = alg(x, outer_iterations=100, inner_iterations=5, trust_region=w, alpha=alpha, gamma=gamma)
    
    @test constr1(result.minimizer) < 1e-2
    @test constr2(result.minimizer) < 1e-2
    @test abs(result.minimum - sqrt(8/27)) < 1e-2
    @test norm(result.minimizer - [1/3, 8/27]) < 1e-2 
end
