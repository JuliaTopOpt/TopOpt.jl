using Revise, FDM, Test
using TopOpt
using TopOpt.AugLag
using Test, LinearAlgebra

using TopOpt.AugLag: Constraint, Objective, LagrangianAlgorithm, AugmentedLagrangianAlgorithm, LagrangianFunction, AugmentedPenalty, IneqConstraintBlock, EqConstraintBlock
using TopOpt.Algorithms: BoxOptimizer

problem = PointLoadCantilever(Val{:Linear}, (2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
solver = FEASolver(Displacement, Direct, problem, xmin = 0.001, penalty = TopOpt.PowerPenalty(1.0))

comp = ComplianceFunction(problem, solver, filtering = false,
    rmin = 4.0, tracing = true, logarithm = false)
x = similar(solver.vars); x .= 0.7;
grad = similar(x)
comp(x, grad)
grad_fdm = FDM.grad(central_fdm(5, 1), x -> comp(x, similar(x)), x)
@test isapprox(grad_fdm, grad, rtol=1e-5)

vol = VolumeFunction(problem, solver)
x = similar(solver.vars); x .= 0.5;
grad = similar(x)
vol(x, grad)
grad_fdm = FDM.grad(central_fdm(5, 1), x -> vol(x, similar(x)), x)
@test isapprox(grad_fdm, grad, rtol=1e-5)

constr = Constraint(VolumeFunction(problem, solver), 0.3)
ineq_block = IneqConstraintBlock((constr,), [100.0], [0.0])
eq_block = EqConstraintBlock((), [], [])
pen = AugmentedPenalty(eq_block, ineq_block, Ref(100.0))
obj = LagrangianFunction(comp, pen)

optimizer = BoxOptimizer(obj)
w = 0.2; gamma=2.0; alpha=0.5
alg = AugmentedLagrangianAlgorithm(optimizer, obj, 20, copy(x), w, gamma, alpha)
AugLag.reset!(alg);
alg.lag.penalty.r[] = 1.0
result = alg(x)

@test constr(result.minimizer) < 1e-3
