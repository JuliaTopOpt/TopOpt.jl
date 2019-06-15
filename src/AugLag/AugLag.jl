module AugLag

using ..TopOpt: @params, TopOpt, dim
using ..TopOpt.Functions: Constraint, AbstractConstraint, AbstractFunction, Objective
using Parameters
using ..TopOpt.Algorithms: setbounds!

abstract type AbstractConstraintBlock end

@params struct EqConstraintBlock <: AbstractConstraintBlock
    block::Union{Function, AbstractArray{<:Function}, Tuple{Vararg{Function}}}
    λ
    grad_λ
end
TopOpt.dim(c::EqConstraintBlock{<:Function}) = 1
TopOpt.dim(c::EqConstraintBlock{<:Tuple}) = sum(dim.(c.block))
TopOpt.dim(c::EqConstraintBlock{<:Array}) = mapreduce(dim, +, c.block, init=0)

@params struct IneqConstraintBlock <: AbstractConstraintBlock
    block::Union{Function, Vector{<:Function}, Tuple{Vararg{Function}}}
    λ
    grad_λ
end
TopOpt.dim(c::IneqConstraintBlock{<:Function}) = 1
TopOpt.dim(c::IneqConstraintBlock{<:Tuple}) = sum(dim.(c.block))
TopOpt.dim(c::IneqConstraintBlock{<:Array}) = mapreduce(dim, +, c.block, init=0)

abstract type AbstractPenalty{T} <: AbstractFunction{T} end

@params struct LinearPenalty{T} <: AbstractPenalty{T}
    eq::EqConstraintBlock
    ineq::IneqConstraintBlock
end
function LinearPenalty(eq::EqConstraintBlock, ineq::IneqConstraintBlock)
    T1 = eltype(eq.λ)
    T2 = eltype(ineq.λ)
    if length(eq.λ) > 0 && length(ineq.λ) > 0
        T = promote_type(T1, T2)
    elseif length(eq.λ) > 0
        T = T1
    elseif length(ineq.λ) > 0
        T = T2
    else
        T = Float64
    end
    LinearPenalty{T, typeof(eq), typeof(ineq)}(eq, ineq)
end

function (d::LinearPenalty)(x, grad_x; reset_grad=true)
    if reset_grad
        grad_x .= 0
        #d.eq.grad_λ .= 0
        #d.ineq.grad_λ .= 0
    end
    v = @views compute_lin_penalty(x, grad_x, d.eq.λ, d.eq.grad_λ, d.eq.block, 0, false)
    v += @views compute_lin_penalty(x, grad_x, d.ineq.λ, d.ineq.grad_λ, d.ineq.block, 0, false)
    return v
end

@inline function compute_lin_penalty(x, grad_x, λ, grad_λ, block::Tuple, offset, reset_grad)
    length(block) === 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_penalty(block[1], x, grad_x, λ[range], grad_λ[range], reset_grad)
    p += compute_lin_penalty(x, grad_x, λ, grad_λ, Base.tail(block), offset+dim(block[1]), false)
    return p
end
@inline function compute_lin_penalty(x, grad_x, λ, grad_λ, block::AbstractArray, offset, reset_grad)
    length(block) == 0 && return zero(eltype(x))
    range =  offset + 1 : offset + dim(block[1])
    p = @views compute_lin_penalty(block[1], x, grad_x, λ[range], grad_λ[range], reset_grad)
    for i in 2:length(block)
        offset += dim(block[i-1])
        range =  offset + 1 : offset + dim(block[i])
        p += @views compute_lin_penalty(block[i], x, grad_x, λ[range], grad_λ[range], false)
    end
    return p
end
# Should be overloaded for block constraints
@inline function compute_lin_penalty(f::AbstractConstraint, x::AbstractArray{<:Real}, grad_x::AbstractArray{<:Real}, λ::AbstractArray{<:Real}, grad_λ::AbstractArray{<:Real}, reset_grad::Bool)
    @assert dim(f) == 1
    if reset_grad
        grad_x .= 0
    end
    v = f(x)
    grad_λ[1] = v
    v *= λ[1]
    grad_x .+= f.grad .* λ[1]
    return v
end

@inline function compute_quad_penalty_ineq(x, grad_x, block::Tuple, r, reset_grad)
    length(block) === 0 && return zero(eltype(x))
    p = compute_quad_penalty_ineq(block[1], x, grad_x, r, reset_grad)
    p += compute_quad_penalty_ineq(x, grad_x, Base.tail(block), r, false)
    return p
end
@inline function compute_quad_penalty_ineq(x, grad_x, block::AbstractArray, r, reset_grad)
    length(block) == 0 && return zero(eltype(x))
    p = compute_quad_penalty_ineq(block[1], x, grad_x, r, reset_grad)
    for i in 2:length(block)
        p += compute_quad_penalty_ineq(block[i], x, grad_x, r, false)
    end
    return p
end
# To be efficiently defined for each block function
function compute_quad_penalty_ineq(f::AbstractConstraint, x::AbstractArray{<:Real}, g::AbstractArray{<:Real}, r, reset_grad::Bool)
    @assert dim(f) == 1
    T = eltype(x)
    if reset_grad
        g .= 0
    end
    v = f(x)
    if v > 0
        g .+= f.grad .* 2*v*r
        v = r*v^2
    else
        v = zero(T)
    end
    return v
end

@inline function compute_quad_penalty_eq(x, grad_x, g::Tuple, r, reset_grad)
    length(g) === 0 && return zero(eltype(x))
    p = compute_quad_penalty_eq(g.block[1], x, grad_x, r, reset_grad)
    p += compute_quad_penalty_eq(x, grad_x, Base.tail(g.block), r, false)
    return p
end
@inline function compute_quad_penalty_eq(x, grad_x, block::AbstractArray, r, reset_grad)
    length(block) == 0 && return zero(eltype(x))
    p = compute_quad_penalty_eq(block[1], x, grad_x, r, reset_grad)
    for i in 2:length(block)
        p += compute_quad_penalty_eq(block[i], x, grad_x, r, false)
    end
    return p
end
# To be efficiently defined for each block function
function compute_quad_penalty_eq(f::AbstractConstraint, x::AbstractArray{<:Real}, g::AbstractArray{<:Real}, r, reset_grad::Bool)
    @assert dim(f) == 1
    if reset_grad
        g .= 0
    end
    v = f(x)
    m = 2*r*v
    v = r*v^2
    g .+= f.grad .* m
    return v
end

@params struct AugmentedPenalty{T} <: AbstractPenalty{T}
    eq::EqConstraintBlock
    ineq::IneqConstraintBlock
    r::Base.RefValue{T}
end

using LinearAlgebra

function (pen::AugmentedPenalty)(x, grad_x; reset_grad = true)
    if reset_grad
        grad_x .= 0
    end
    @unpack eq, ineq, r = pen
    p = @views compute_lin_penalty(x, grad_x, eq.λ, eq.grad_λ, eq.block, 0, false)
    p += @views compute_quad_penalty_eq(x, grad_x, eq.block, r[], false)
    p += @views compute_lin_penalty(x, grad_x, ineq.λ, ineq.grad_λ, ineq.block, 0, false)
    p += @views compute_quad_penalty_ineq(x, grad_x, ineq.block, r[], false)
    return p
end

@params struct LagrangianFunction{T} <: AbstractFunction{T}
    obj::AbstractFunction{T}
    penalty::AbstractPenalty
    grad::AbstractVector{T}
end
LagrangianFunction(obj, penalty) = LagrangianFunction(obj, penalty, similar(obj.grad))

function Base.getproperty(func::LagrangianFunction, f::Symbol)
    f === :obj && return getfield(func, :obj)
    f === :penalty && return getfield(func, :penalty)
    f === :grad && return getfield(func, :grad)
    return getproperty(getfield(func, :obj), f)
end
function (l::LagrangianFunction)(x, grad=l.grad)
    @unpack obj, penalty = l
    grad .= 0
    p = obj(x, grad)
    p += penalty(x, grad, reset_grad=false)
    if grad !== l.grad
        l.grad .= grad
    end
    return p
end
const AugmentedLagrangianFunction = LagrangianFunction{<:Any, <:Any, <:AugmentedPenalty}

abstract type AbstractLagrangianAlgorithm end

@params struct LagrangianAlgorithm <: AbstractLagrangianAlgorithm
    optimizer
    lag
    n::Int
    x
    prev_grad
    w
    α
end
function LagrangianAlgorithm(optimizer, lag::LagrangianFunction, n::Int, x, w = 0.25, α = 0.1)
    prev_grad = similar(x)
    return LagrangianAlgorithm(optimizer, lag, n, x, prev_grad, w, α)
end

@params struct AugmentedLagrangianAlgorithm <: AbstractLagrangianAlgorithm
    optimizer
    lag
    n::Int
    x
    prev_grad
    w
    γ
    α
end
function AugmentedLagrangianAlgorithm(optimizer, lag::AugmentedLagrangianFunction, n::Int, x, w = 0.25, γ = 1.2, α = 0.1)
    prev_grad = similar(x)
    return AugmentedLagrangianAlgorithm(optimizer, lag, n, x, prev_grad, w, γ, α)
end

function reset!(alg::AbstractLagrangianAlgorithm)
    @unpack eq, ineq = alg.lag.penalty
    if length(eq.λ) > 0
        eq.λ .= 0
    end
    if length(ineq.λ) > 0
        ineq.λ .= 0
    end
    alg.prev_grad .= 0
    if alg isa AugmentedLagrangianAlgorithm
        alg.lag.penalty.r[] = 1.0
    end
end

for Talg in (:AugmentedLagrangianAlgorithm, :LagrangianAlgorithm)
    @eval begin
        function (alg::$Talg)(x0)
            @unpack optimizer, lag, n, x, prev_grad, w, α = alg
            @unpack eq, ineq = lag.penalty
            if $Talg <: AugmentedLagrangianAlgorithm
                @unpack γ = alg
                @unpack r = lag.penalty
            end
            T = eltype(x0)
            x .= x0
            setbounds!(optimizer, x, w)
            result = optimizer(x)
            x .= result.minimizer
            prev_grad .= lag.grad
            prev_l = result.minimum
            setbounds!(optimizer, x, w)
            for i in 2:n
                if length(eq.λ) > 0
                    eq.λ .+= α .* eq.grad_λ
                end
                if length(ineq.λ) > 0
                    ineq.λ .+= max.(α .* ineq.grad_λ, zero(T))
                end
                if $Talg <: AugmentedLagrangianAlgorithm
                    r[] *= γ
                end
                result = optimizer(x)
                w, update = getbounds(w, prev_grad, x, result.minimizer, prev_l, result.minimum)
                if update
                    x .= result.minimizer
                    prev_grad .= lag.grad
                    prev_l = result.minimum
                end
                setbounds!(optimizer, x, w)
            end
            return result
        end
    end
end

function getbounds(w, prev_grad, x_prev, x, prev_l, l)
    mu = (l - prev_l) / (dot(prev_grad, x) - dot(prev_grad, x_prev))
    if mu <= 0.25
        return w / 4, false
    elseif 0.25 < mu <= 0.75
        return w, true
    else
        return min(2*w, 1.0), true
    end
end

end
