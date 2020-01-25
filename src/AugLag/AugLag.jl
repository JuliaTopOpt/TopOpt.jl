module AugLag

using FillArrays, CatViews, Optim, LineSearches, LinearAlgebra
using ..TopOpt: @params, TopOpt, dim, PENALTY_BEFORE_INTERPOLATION
using ..TopOpt.Functions: Constraint, AbstractConstraint, AbstractFunction, Objective, LinAggregation, LinQuadAggregation
using Parameters, Statistics
using ..TopOpt.Algorithms: setbounds!
using ..TopOpt.Utilities: Utilities, setpenalty!, @forward_property
using ..TopOpt.Optimise

include("optimizers.jl")

abstract type AbstractConstraintBlock end

for blockT in (:EqConstraintBlock, :IneqConstraintBlock)
    @eval begin
        @params struct $blockT <: AbstractConstraintBlock
            block::Union{Function, AbstractArray{<:Function}, Tuple{Vararg{Function}}}
            λ
            grad_λ
        end
        TopOpt.dim(c::$blockT{<:Function}) = 1
        TopOpt.dim(c::$blockT{<:Tuple}) = sum(dim.(c.block))
        TopOpt.dim(c::$blockT{<:Array}) = mapreduce(dim, +, c.block, init=0)
    end
end

abstract type AbstractPenalty{T} <: AbstractFunction{T} end

## Linear penalty ##
@params struct LinearPenalty{T} <: AbstractPenalty{T}
    eq::EqConstraintBlock
    ineq::IneqConstraintBlock
    grad_temp::AbstractVector{T}
end
function LinearPenalty(eq::EqConstraintBlock, ineq::IneqConstraintBlock)
    if length(eq.block) > 0
        grad_temp = similar(eq.block[1].grad)
    elseif length(ineq.block) > 0
        grad_temp = similar(ineq.block[1].grad)
    else
        grad_temp = Float64[]
    end
    LinearPenalty(eq, ineq, grad_temp)
end

function (d::LinearPenalty)(x, grad_x; reset_grad=true)
    if reset_grad
        grad_x .= 0
    end
    @unpack eq, ineq, grad_temp = d
    v = @views compute_lin_penalty(x, grad_x, eq.λ, eq.grad_λ, eq.block, grad_temp, 0)
    v += @views compute_lin_penalty(x, grad_x, ineq.λ, ineq.grad_λ, ineq.block, grad_temp, 0)
    return v
end

@inline function compute_lin_penalty(x, grad_x, λ, grad_λ, block::Tuple, grad_temp, offset)
    length(block) === 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_penalty(block[1], x, grad_x, λ[range], grad_λ[range], grad_temp)
    p += compute_lin_penalty(x, grad_x, λ, grad_λ, Base.tail(block), grad_temp, offset + dim(block[1]))
    return p
end
@inline function compute_lin_penalty(x, grad_x, λ, grad_λ, block::AbstractArray{<:Function}, grad_temp, offset)
    length(block) == 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_penalty(block[1], x, grad_x, λ[range], grad_λ[range], grad_temp)
    for i in 2:length(block)
        offset += dim(block[i-1])
        range =  offset + 1 : offset + dim(block[i])
        p += @views compute_lin_penalty(block[i], x, grad_x, λ[range], grad_λ[range], grad_temp)
    end
    return p
end
# Should be overloaded for block constraints
@inline function compute_lin_penalty(f::AbstractConstraint, x, grad_x, λ, grad_λ, grad_temp)
    func1 = LinAggregation(f, grad_λ, λ, grad_temp, 0, 1)
    v = func1(x)
    grad_x .+= func1.grad
    return v
end

## Augmented penalty ##

@params struct AugmentedPenalty{T} <: AbstractPenalty{T}
    eq::EqConstraintBlock
    ineq::IneqConstraintBlock
    r::Base.RefValue{T}
    grad_temp::AbstractVector{T}
end
function AugmentedPenalty(eq::EqConstraintBlock, ineq::IneqConstraintBlock, r::T) where {T}
    @assert !(typeof(r) <: Ref)
    if length(eq.block) > 0
        grad_temp = similar(eq.block[1].grad)
    elseif length(ineq.block) > 0
        grad_temp = similar(ineq.block[1].grad)
    else
        grad_temp = T[]
    end
    return AugmentedPenalty(eq, ineq, Ref{eltype(grad_temp)}(r), grad_temp)
end

function (pen::AugmentedPenalty)(x, grad_x; reset_grad = true)
    if reset_grad
        grad_x .= 0
    end
    @unpack eq, ineq, r, grad_temp = pen
    p = compute_lin_quad_penalty(x, grad_x, eq.λ, eq.grad_λ, eq.block, r[], grad_temp, 0, true)
    p += compute_lin_quad_penalty(x, grad_x, ineq.λ, ineq.grad_λ, ineq.block, r[], grad_temp, 0, false)
    return p
end

@inline function compute_lin_quad_penalty(x, grad_x, λ, grad_λ, block::Tuple, r, grad_temp, offset, equality)
    length(block) === 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_quad_penalty(block[1], x, grad_x, λ[range], grad_λ[range], r, grad_temp, equality)
    p += compute_lin_quad_penalty(x, grad_x, λ, grad_λ, Base.tail(block), r, grad_temp, offset + dim(block[1]), equality)
    return p
end
@inline function compute_lin_quad_penalty(x, grad_x, block::AbstractArray{<:Function}, r, grad_temp, equality)
    length(block) == 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_quad_penalty(block[1], x, grad_x, λ[range], grad_λ[range], r, grad_temp, equality)
    for i in 2:length(block)
        offset += dim(block[i-1])
        range =  offset + 1 : offset + dim(block[i])
        p += @views compute_lin_quad_penalty(block[i], x, grad_x, λ[range], grad_λ[range], r, grad_temp, equality)
    end
    return p
end
@inline function compute_lin_quad_penalty(f::AbstractConstraint, x, grad_x, λ, grad_λ, r, grad_temp, equality)
    func = LinQuadAggregation(f, grad_λ, λ, r, !equality, grad_temp, 0, 1)
    func.grad .= 0
    v = func(x)
    grad_x .+= func.grad
    return v
end

## Lagrangian function ##
@params mutable struct Lagrangian{T} <: AbstractFunction{T}
    obj::AbstractFunction{T}
    penalty::AbstractPenalty
    objval::T
    penaltyval::T
    violation::T
    grad::AbstractVector{T}
end
function Lagrangian(obj, penalty)
    T = eltype(obj.grad)
    return Lagrangian(obj, penalty, zero(T), zero(T), zero(T), similar(obj.grad))
end
function Lagrangian(::Type{PenT}, obj; eq=(), ineq = (), r0 = 1.0, λ0 = 1.0) where {PenT}
	if length(ineq) > 0
		n = sum(TopOpt.dim.(ineq))
		λ = similar(obj.grad, n)
		grad_λ = similar(λ)
		λ .= λ0
		grad_λ .= 0
	else
		λ = []
		grad_λ = []
	end
	ineq_block = IneqConstraintBlock(ineq, λ, grad_λ)
	if length(eq) > 0
		n = sum(TopOpt.dim.(eq))
		λ = similar(obj.grad, n)
		grad_λ = similar(λ)
		λ .= λ0
		grad_lambda .= 0
	else
		λ = []
		grad_λ = []
	end
	eq_block = EqConstraintBlock(eq, λ, grad_λ)
	if PenT <: AugmentedPenalty
		pen = AugmentedPenalty(eq_block, ineq_block, r0)
	elseif PenT <: LinearPenalty
		pen = LinearPenalty(eq_block, ineq_block)
	else
		throw("Unsupported penalty type $PenT.")
	end
	return Lagrangian(obj, pen)
end

TopOpt.dim(l::Lagrangian) = 1

function Utilities.setpenalty!(lag::Lagrangian, p)
    setpenalty!(lag.obj, p)
    for c in lag.penalty.eq.block
        setpenalty!(c, p)
    end
    for c in lag.penalty.ineq.block
        setpenalty!(c, p)
    end
    return lag
end
@forward_property Lagrangian obj

function (l::Lagrangian)(x, grad=l.grad)
    @unpack obj, penalty = l
    @unpack eq, ineq = penalty
    T = eltype(grad)
    grad .= 0
    p1 = obj(x, grad)
    l.objval = p1
    p2 = penalty(x, grad, reset_grad=false)
    l.penaltyval = p2
    max_eq_infeas = length(eq.grad_λ) == 0 ? zero(T) : maximum(abs, eq.grad_λ)
    max_ineq_infeas = length(ineq.grad_λ) == 0 ? zero(T) : maximum(x -> max(0, x), ineq.grad_λ)
    l.violation = max(max_eq_infeas, max_ineq_infeas)

    if grad !== l.grad
        l.grad .= grad
    end
    return p1 + p2
end
const AugmentedLagrangian = Lagrangian{<:Any, <:Any, <:AugmentedPenalty}

## Lagrangian algorithm ##

abstract type AbstractLagrangianAlgorithm end

@params struct LagrangianAlgorithm <: AbstractLagrangianAlgorithm
    optimizer
    lag
    x
    prev_grad
end
function LagrangianAlgorithm(optimizer, lag::Lagrangian, n::Int, x)
    prev_grad = similar(x)
    return LagrangianAlgorithm(optimizer, lag, x, prev_grad)
end

@params struct AugmentedLagrangianAlgorithm <: AbstractLagrangianAlgorithm
    optimizer
    lag
    x
    prev_grad
end
function AugmentedLagrangianAlgorithm(optimizer, lag::AugmentedLagrangian, x)
    prev_grad = similar(x)
    return AugmentedLagrangianAlgorithm(optimizer, lag, x, prev_grad)
end

function reset!(alg::AbstractLagrangianAlgorithm; r = NaN, λ = NaN, x = alg.x)
    @unpack eq, ineq = alg.lag.penalty
    if length(eq.λ) > 0 && isfinite(λ)
        eq.λ .= λ
    end
    if length(ineq.λ) > 0 && isfinite(λ)
        ineq.λ .= λ
    end
    alg.prev_grad .= 0
    if alg isa AugmentedLagrangianAlgorithm && isfinite(r)
        alg.lag.penalty.r[] = r
    end
    alg.x .= x
    return alg
end

@params struct AugLagResult{T}
    minimizer::AbstractVector{T}
    minimum::T
    fevals::Int
end

macro print_verbose()
    esc(quote
        println("    Lagrangian min = $(lag(sol.primal.x))")
        println("    Obj = $(lag.objval)\n")
        println("    Gradient inf-norm = $(maximum(abs, lag.grad))")
        max_eq_infeas = length(eq.grad_λ) == 0 ? 0.0 : maximum(abs, eq.grad_λ)
        max_ineq_infeas = length(ineq.grad_λ) == 0 ? 0.0 : maximum(x -> max(0, x), ineq.grad_λ)
        println("    Max eq infeasibility = $(max_eq_infeas)")
        println("    Max ineq infeasibility = $(max_ineq_infeas)\n")
    end)
end

function catview(v1, v2)
    if length(v1) == 0 && length(v2) != 0
        return CatView(v2)
    elseif length(v1) != 0 && length(v2) == 0
        return CatView(v1)
    elseif length(v1) != 0 && length(v2) != 0
        return CatView(v1, v2)
    else
        T1, T2 = eltype(v1), eltype(v2)
        return CatView(promote_type(T1, T2)[])
    end
end

@params struct Solution
    lagval
    objval
    penaltyval
    violation
    feasible
    primal
    dual
end
@params struct PrimalSolution
    x
end
@params struct DualSolution
    ineq_λ
    eq_λ
end
function InitialSolution(lag, x0)
    ineq_λ = copy(lag.penalty.ineq.λ)
    eq_λ = copy(lag.penalty.eq.λ)
    lagval = lag(x0)
    primal = PrimalSolution(copy(x0))
    dual = DualSolution(copy(ineq_λ), copy(eq_λ))
    return Solution(lagval, lag.objval, lag.penaltyval, lag.violation, lag.violation == 0, primal, dual)
end
function Solution(lag, sol::Solution)
    x = sol.primal.x
    ineq_λ, eq_λ = sol.dual.ineq_λ, sol.dual.eq_λ
    lag.penalty.ineq.λ .= ineq_λ
    lag.penalty.eq.λ .= eq_λ
    lagval = lag(x)
    primal = PrimalSolution(copy(x))
    dual = DualSolution(copy(ineq_λ), copy(eq_λ))
    return Solution(lagval, lag.objval, lag.penaltyval, lag.violation, lag.violation == 0, primal, dual)
end

@params struct PrimalTransition
    grad # gradient of the Lagrangian
    cg # conjugate gradient
    line # projected
    step # step according to CG
    islarge::Bool # large enough step or no
end
function primaltransition(opt::Union{CG, LBFGS}, sol::Solution, prev_trans::Union{PrimalTransition, Nothing}, lag, lb, ub, alpha0, nhalves, xtol, ftol, infeastol, use_cg, func_evals, verbose, factor)
    initial = prev_trans isa Nothing
    prev_sol = deepcopy(sol)
    length(sol.primal.x) == 0 && throw("Empty primal solution.")
    T = eltype(sol.primal.x)
    prev_lagval = sol.lagval
    prev_grad = initial ? lag.grad : prev_trans.grad
    prev_cg = initial ? lag.grad : prev_trans.cg
    grad = lag.grad
    x = sol.primal.x
    if opt isa LBFGS
        line = opt.invH * grad
        update!(opt.invH, grad, x)
    else
        if use_cg
            line = getcg(grad, prev_grad, prev_cg, initial)
        else
            line = grad
        end
    end
    step = alpha0
    islarge = maximum(1:length(line)) do i
        line[i] > 0 && return min(step * line[i], x[i] - lb[i])
        line[i] < 0 && return min(-step * line[i], ub[i] - x[i])
        return zero(T)
    end > xtol
    prev_x = copy(sol.primal.x)
    if islarge
        sol.primal.x .= clamp.(sol.primal.x .- step .* line, lb, ub)
        sol = Solution(lag, sol)
        func_evals += 1
        i = 1
        while sol.lagval > prev_lagval && i <= nhalves
            step *= factor
            sol.primal.x .= clamp.(prev_x .- step .* line, lb, ub)
            sol = Solution(lag, sol)
            func_evals += 1
            i += 1
        end
    end
    trans = PrimalTransition(copy(grad), line, line, step, islarge)

    verbose && println("    x_step = $(step), min(line) = $(minimum(line)), max(line) = $(maximum(line))")
    objchange = abs(sol.objval - prev_sol.objval) / (abs(prev_sol.objval) + ftol)
    penaltychange =  abs(sol.penaltyval - prev_sol.penaltyval) / (abs(prev_sol.penaltyval) + ftol)
    converged = objchange <= ftol && (penaltychange <= ftol || sol.violation <= infeastol)
    converged = ipopt_converged(sol.primal.x, lb, ub, lag.grad, sol.dual.ineq_λ, sol.dual.eq_λ, sol.violation, ftol)

    return sol, trans, converged, func_evals
end

function ipopt_converged(x, lb, ub, lag_grad, ineq_λ, eq_λ, violation, tol)
    T = eltype(lag_grad)
    z = map(1:length(lag_grad)) do i 
        if x[i] <= lb[i] && lag_grad[i] > 0
            return -(lag_grad[i])
        elseif x[i] >= ub[i] && lag_grad[i] < 0
            return lag_grad[i]
        else
            return zero(T)
        end                
    end
    smax = 100
    s = zero(T)
    if length(ineq_λ) != 0
        s += sum(abs, ineq_λ)
    end
    if length(eq_λ) != 0
        s += sum(abs, eq_λ)
    end
    s += sum(abs, z)
    n = length(ineq_λ) + length(eq_λ) + length(lag_grad)
    sd = max(smax, s/n)/smax
    residual = max(norm(lag_grad + z, Inf)/sd, violation)
    @show residual
    return residual <= tol
end

function primaltransition(sgd::FluxOpt, sol::Solution, prev_trans::Union{PrimalTransition, Nothing}, lag, lb, ub, alpha0, nhalves, xtol, ftol, infeastol, use_cg, func_evals, verbose, factor)
    initial = prev_trans isa Nothing
    prev_sol = deepcopy(sol)
    prev_x = prev_sol.primal.x
    prev_lagval = prev_sol.lagval
    length(sol.primal.x) == 0 && throw("Empty primal solution.")
    T = eltype(sol.primal.x)
    grad = lag.grad
    line = copy(grad)
    Optimise.apply!(sgd, sol.primal.x, line)
    sol.primal.x .= clamp.(sol.primal.x .- alpha0 .* line, lb, ub)
    sol = Solution(lag, sol)
    func_evals += 1
    trans = PrimalTransition(copy(lag.grad), copy(lag.grad), line, alpha0, true)

    verbose && println("    norm(x - prev_x) = $(norm(sol.primal.x .- prev_x)), min(line) = $(minimum(line)), max(line) = $(maximum(line))")
    objchange = abs(sol.objval - prev_sol.objval) / (abs(prev_sol.objval) + ftol)
    penaltychange =  abs(sol.penaltyval - prev_sol.penaltyval) / (abs(prev_sol.penaltyval) + ftol)
    converged = objchange <= ftol && (penaltychange <= ftol || sol.violation <= infeastol)

    return sol, trans, converged, func_evals
end

@params struct DualTransition
    ineq_grad
    ineq_cg
    ineq_line
    eq_grad
    eq_cg
    eq_line
    ineq_step
    eq_step
    islarge
end
function dualtransition(opt::Union{CG, LBFGS}, sol::Solution, prev_trans::Union{DualTransition, Nothing}, lag, alpha0, nhalves, λtol, ftol, use_cg, dual_margin, func_evals, verbose, factor)
    initial = prev_trans isa Nothing
    prev_sol = deepcopy(sol)
    length(sol.primal.x) == 0 && throw("Empty primal solution.")
    length(sol.dual.ineq_λ) == 0 && length(sol.dual.eq_λ) == 0 && throw("Empty dual solution.")
    T = eltype(sol.primal.x)
    use_cg = use_cg && opt isa CG

    prev_lagval = sol.lagval

    prev_ineq_grad = initial ? lag.penalty.ineq.grad_λ : prev_trans.ineq_grad
    prev_ineq_cg = initial ? lag.penalty.ineq.grad_λ : prev_trans.ineq_cg

    prev_eq_grad = initial ? lag.penalty.eq.grad_λ : prev_trans.eq_grad
    prev_eq_cg = initial ? lag.penalty.eq.grad_λ : prev_trans.eq_cg

    ineq_grad = lag.penalty.ineq.grad_λ
    if use_cg
        ineq_cg = getcg(ineq_grad, prev_ineq_grad, prev_ineq_cg, initial)
    else
        ineq_cg = ineq_grad
    end
    ineq_line = ineq_cg
    #ineq_line, ineq_step, ineq_islarge = getlineandstep(sol.dual.ineq_λ, ineq_cg, dual_margin, Inf, λtol, false)

    eq_grad = lag.penalty.eq.grad_λ
    if use_cg
        eq_cg = getcg(eq_grad, prev_eq_grad, prev_eq_cg, initial)
    else
        eq_cg = eq_grad
    end
    eq_line = eq_cg
    #eq_line, eq_step, eq_islarge = getlineandstep(sol.dual.eq_λ, eq_cg, -Inf, Inf, λtol, false)

    ineq_step = eq_step = alpha0
    #eq_step = min(alpha0, eq_step)
    #ineq_step = min(alpha0, ineq_step)

    #islarge = eq_islarge || ineq_islarge
    islarge = true
    prev_ineq_λ = copy(sol.dual.ineq_λ)
    prev_eq_λ = copy(sol.dual.eq_λ)
    if islarge
        sol.dual.ineq_λ .= max.(prev_ineq_λ .+ ineq_step .* ineq_line, dual_margin)
        sol.dual.eq_λ .= prev_eq_λ .+ eq_step .* eq_line
        sol = Solution(lag, sol)
        func_evals += 1
        i = 1
        while sol.lagval < prev_lagval && i <= nhalves
            eq_step *= factor
            ineq_step *= factor
            sol.dual.ineq_λ .= max.(prev_ineq_λ .+ ineq_step .* ineq_line, dual_margin)
            sol.dual.eq_λ .= prev_eq_λ .+ eq_step .* eq_line
            sol = Solution(lag, sol)
            func_evals += 1
            i += 1
        end
    end
    trans = DualTransition(copy(ineq_grad), copy(ineq_cg), copy(ineq_line), copy(eq_grad), copy(eq_cg), copy(eq_line), ineq_step, eq_step, islarge)

    verbose && println("ineq_step = $(ineq_step), maximum(ineq_λ) = $(maximum(sol.dual.ineq_λ)), minimum(ineq_λ) = $(minimum(sol.dual.ineq_λ)), maximum(ineq_grad) = $(maximum(ineq_grad)), minimum(ineq_grad) = $(minimum(ineq_grad)), maximum(ineq_line) = $(maximum(ineq_line)), minimum(ineq_line) = $(minimum(ineq_line))")
    verbose && println("eq_step = $(eq_step), eq = $(sol.dual.eq_λ), eq_grad = $(eq_grad), eq_line = $(eq_line)")

    penaltychange =  abs(sol.penaltyval - prev_sol.penaltyval) / (abs(prev_sol.penaltyval) + ftol)
    converged = penaltychange <= ftol
    #converged = ipopt_converged(sol.primal.x, lb, ub, lag.grad, sol.dual.ineq_λ, sol.dual.eq_λ, sol.violation, ftol)

    return sol, trans, converged, func_evals
end

for Talg in (:AugmentedLagrangianAlgorithm, :LagrangianAlgorithm)
    @eval begin
        function (alg::$Talg)(x0=alg.x; verbose=true, outer_iterations=5, inner_iterations=20, primal_alpha0=1.0, dual_alpha0=1.0, gamma=3.0, trust_region=1.0, ftol=1e-5, xtol=1e-4, λtol=1e-3, infeastol=1e-5, primal_nhalves=10, dual_nhalves=10, adapt_trust_region=false, primal_cg=false, dual_cg=false, dual_margin=0.0, primal_optim = LBFGS(4), dual_optim = CG(), adapt_primal_step=1, adapt_dual_step=1, primal_step_adapt_factor0=2.0, dual_step_adapt_factor0=2.0)
            @unpack optimizer, lag, x, prev_grad = alg
            @unpack eq, ineq = lag.penalty
            if $Talg <: AugmentedLagrangianAlgorithm
                @unpack r = lag.penalty
            end
            T = eltype(x0)
            x .= x0
            if primal_optim isa LBFGS
                n = primal_optim.n
                primal_optim = LBFGS(n, ApproxInverseHessian(x, n))
            end
            primal_step_adapt_factor = primal_step_adapt_factor0
            dual_step_adapt_factor = dual_step_adapt_factor0
            sol = InitialSolution(lag, x0)
            best = deepcopy(sol)
            setbounds!(optimizer, x, trust_region)
            func_evals = 0
            tobreak = false
            #if $Talg <: AugmentedLagrangianAlgorithm
            #    dual_alpha = r[]
            #else
                dual_alpha = dual_alpha0
            #end
            primal_trans = dual_trans = nothing
            primal_converged = dual_converged = false
            verbose && println("maximum(Initial ineq_λ) = $(maximum(sol.dual.ineq_λ)), minimum(Initial ineq_λ) = $(minimum(sol.dual.ineq_λ)), maximum(initial ineq_grad) = $(maximum(lag.penalty.ineq.grad_λ)), minimum(initial ineq_grad) = $(minimum(lag.penalty.ineq.grad_λ))")
            for i in 1:outer_iterations
                primal_alpha = primal_alpha0
                for j in 1:inner_iterations
                    prev_grad = primal_trans isa Nothing ? lag.grad : copy(primal_trans.grad)
                    prev_x = copy(sol.primal.x)
                    prev_lagval = sol.lagval
                    sol, primal_trans, primal_converged, func_evals = primaltransition(primal_optim, sol, primal_trans, lag, optimizer.lb, optimizer.ub, primal_alpha, primal_nhalves, xtol, ftol, infeastol, primal_cg, func_evals, true, 1/primal_step_adapt_factor)
                    if !(primal_optim isa FluxOpt) && adapt_primal_step == 1
                        primal_alpha = min(primal_alpha0, 2*primal_trans.step)
                    elseif !(primal_optim isa FluxOpt) && adapt_primal_step == 2
                        primal_alpha = primal_step_adapt_factor*primal_trans.step
                        #=
                        limit = trust_region / mean(abs, primal_trans.line)
                        new_alpha = primal_step_adapt_factor * primal_trans.step
                        if limit <= new_alpha
                            primal_alpha = limit
                            primal_step_adapt_factor = max(primal_step_adapt_factor * 0.9, 1.05)
                        else
                            primal_alpha = new_alpha
                            primal_step_adapt_factor *= 1.1
                        end
                        @show primal_step_adapt_factor
                        =#
                    end
                    setbounds!(optimizer, sol.primal.x, trust_region)
                    if primal_trans.islarge
                        verbose && println("    Current solution: min = $(minimum(sol.primal.x)), max = $(maximum(sol.primal.x)), obj = $(sol.objval), violation = $(sol.violation)")
                        # Best solution
                        if sol.violation < best.violation || sol.feasible && sol.objval <= best.objval
                            best = deepcopy(sol)
                            verbose && println("    New solution found, min = $(minimum(sol.primal.x)), max = $(maximum(sol.primal.x)), violation = $(sol.violation)")
                        # Allowing increase is to exit local minima
                        else
                            verbose && println("    Lagrangian value may have increased, new value = $(sol.lagval)")
                        end
                        verbose && @print_verbose()
                    else
                        break
                    end
                    if j > 1 && primal_converged
                        println("    Breaking primal early: objval = $(sol.objval), penalty = $(sol.penaltyval).")
                        break
                    end
                    if adapt_trust_region
                        trust_region = getbounds(trust_region, prev_grad, sol.primal.x, prev_x, prev_lagval, sol.lagval, xtol)
                        println("    New trust region = $trust_region")
                    end
                end
                if dual_converged && primal_converged
                    println("    Breaking dual early: objval = $(sol.objval), penalty = $(sol.penaltyval).")
                    break
                end
                sol, dual_trans, dual_converged, func_evals = dualtransition(dual_optim, sol, dual_trans, lag, dual_alpha, dual_nhalves, λtol, ftol, dual_cg, dual_margin, func_evals, true, 1/dual_step_adapt_factor0)
                # Adapt the maximum step size
                if adapt_dual_step == 1
                    dual_alpha = min(dual_alpha0, 2*max(dual_trans.ineq_step, dual_trans.eq_step))
                elseif adapt_dual_step == 2
                    dual_alpha = dual_step_adapt_factor0*max(dual_trans.ineq_step, dual_trans.eq_step)
                end
                if $Talg <: AugmentedLagrangianAlgorithm
                    r[] *= gamma
                #    dual_alpha = r[]
                    verbose && println("r = $(r[])")
                end
                dual_converged = false
                #if primal_optim isa LBFGS
                #    reset!(primal_optim.invH)
                #end
            end
            return AugLagResult(best.primal.x, best.objval, func_evals)
        end
    end
end

function getlineandstep(x, grad, _lb, _ub, tol, primal)
    T = eltype(grad)
    s = primal ? -1 : 1
    lb = _lb isa Number ? fill(_lb, length(x)) : _lb
    ub = _ub isa Number ? fill(_ub, length(x)) : _ub
    length(x) == 0 && return grad, zero(T <: Any ? Int : T), false
    line = map(1:length(grad)) do i
        if x[i] <= lb[i] + tol && s*grad[i] < 0 || x[i] >= ub[i] - tol && s*grad[i] > 0
            zero(T)
        else
            grad[i]
        end
    end
    step = minimum(1:length(x)) do i
        if s*line[i] > 0
            s*(ub[i] - x[i]) / line[i]
        elseif s*line[i] < 0 
            -s*(x[i] - lb[i]) / line[i]
        else
            T(Inf)
        end
    end
    maxstep = maximum(step*abs(line[i]) for i in 1:length(line))
    return line, step, maxstep >= tol
end

function getcg(grad, prev_grad, prev_cg, initial)
    length(grad) == 0 && return grad
    @assert all(isfinite, grad)
    initial && return copy(grad)
    #beta = dot(grad, grad)/dot(prev_grad, prev_grad)
    #beta = dot(grad, grad - prev_grad)/dot(prev_grad, prev_grad)
    beta = dot(grad, grad - prev_grad)/dot(prev_cg, grad - prev_grad)
    #beta = dot(grad, grad)/dot(prev_cg, grad - prev_grad)
    isfinite(beta) || return copy(grad)
    return grad .+ beta .* prev_cg
end

function getbounds(trust_region, prev_grad, x_prev, x, prev_l, l, xtol)
    mu = (prev_l - l) / (dot(prev_grad, x) - dot(prev_grad, x_prev))
    @show mu
    if mu <= 0.70
        trust_region = trust_region / 4
    elseif 0.70 < mu <= 0.90
        trust_region = trust_region
    else
        trust_region = min(2*trust_region, 1.0)
    end
    return max(trust_region, 2*xtol)
end

end
