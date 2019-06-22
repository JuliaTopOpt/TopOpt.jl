using Revise, FDM, Test
using TopOpt
using TopOpt.AugLag
using Test, LinearAlgebra

using TopOpt.AugLag: Constraint, Objective, LagrangianAlgorithm, AugmentedLagrangianAlgorithm, LagrangianFunction, AugmentedPenalty, IneqConstraintBlock, EqConstraintBlock
using TopOpt.Algorithms: BoxOptimizer

@testset "Compliance" begin
    function solve()
        local result
        problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(Displacement, Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(1.0))

        comp = ComplianceFunction(problem, solver, filtering = false, rmin = 4.0,
            tracing = true, logarithm = false)
        bin_pen = BinPenaltyFunction(solver, 1.0)
        obj = comp + bin_pen

        constr = Constraint(VolumeFunction(problem, solver), 0.5)
        ineq_block = IneqConstraintBlock((constr,), [1.0e3], [0.0])
        eq_block = EqConstraintBlock((), [], [])
        pen = AugmentedPenalty(eq_block, ineq_block, 1.0)

        lag = LagrangianFunction(obj, pen)
        x = similar(solver.vars); x .= 0.5;
        optimizer = BoxOptimizer(lag, Optim.LBFGS(), options=Optim.Options(allow_outer_f_increases=false, x_tol=1e-5, f_tol=1e-3, g_tol=1e-2, outer_iterations=10))

        alg = AugmentedLagrangianAlgorithm(optimizer, lag, copy(x))
        AugLag.reset!(alg); alg.lag.penalty.ineq.λ[1] = 1.0; alg.lag.penalty.r[] = 1.0
        alg.x .= x
        bin_pen.s = 1e-3
        w = 1.0; gamma=1.05; alpha=500.0
        for i in 1:3
            result = alg(copy(alg.x), verbose=true, outer_iterations=2, alpha=alpha, gamma=gamma, trust_region=w)
            bin_pen.s *= 10
            alpha /= 2
        end
        return result, problem
    end
    result, problem = solve()
end

#=
w = 1.0; gamma=1.05; alpha=5.0
alg2 = AugmentedLagrangianAlgorithm(optimizer, obj, 50, copy(result.minimizer), w, gamma, alpha)
AugLag.reset!(alg2); alg2.lag.penalty.ineq.λ[1] = alg.lag.penalty.ineq.λ[1]; alg.lag.penalty.r[] = 1.0
result = alg(x, verbose=true)
=#

#=
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
=#
