using MMA
using Test, LinearAlgebra

import MMA: eval_constraint, eval_objective, dim

function f(x::AbstractVector, grad::AbstractVector)
    if length(grad) != 0
        grad[1] = 0.0
        grad[2] = 0.5/sqrt(x[2])
    end
    sqrt(x[2])
end

function g(x::AbstractVector, grad::AbstractVector, a, b)
    if length(grad) != 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

ndim = 2
m = MMAModel(ndim, f, xtol = 1e-6, store_trace=true)

box!(m, 1, 0.0, 10.0)
box!(m, 2, 0.0, 10.0)

ineq_constraint!(m, (x,grad) -> g(x,grad,2,0))
ineq_constraint!(m, (x,grad) -> g(x,grad,-1,1))


@test dim(m) == 2

# Objective
let
    grad1 = zeros(2)
    grad2 = zeros(2)
    p = [1.234, 2.345]
    @test eval_objective(m, p, grad2) ≈ f(p, grad1)
    @test grad1 ≈ grad2
end

# Box
@test min(m, 1) == 0.0
@test max(m, 1) == 10.0
@test min(m, 2) == 0.0
@test max(m, 2) == 10.0

# Inequalities
let
    grad1 = zeros(2)
    grad2 = zeros(2)
    p = [1.234, 2.345]
    @test eval_constraint(m, 1, p, grad1) ≈ g(p,grad2 ,2,0)
    @test grad1 ≈ grad2

    @test eval_constraint(m, 2, p, grad1) ≈ g(p,grad2,-1,1)
    @test grad1 ≈ grad2
end

r = optimize(m, [1.234, 2.345], MMA.MMA02())
@test abs(r.minimum - sqrt(8/27)) < 1e-6
@test norm(r.minimizer - [1/3, 8/27]) < 1e-6

r = optimize(m, [1.234, 2.345], MMA.MMA87())
@test abs(r.minimum - sqrt(8/27)) < 1e-6
@test norm(r.minimizer - [1/3, 8/27]) < 1e-6
