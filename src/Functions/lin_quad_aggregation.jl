abstract type AbstractAggregation{T} <: AbstractFunction{T} end
@inline function Base.getproperty(vf::AbstractAggregation, f::Symbol)
    f === :reuse && return vf.f.reuse
    f === :solver && return vf.f.solver
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::AbstractAggregation, f::Symbol, v)
    f === :reuse && return setproperty!(vf.f, f, v)
    return setfield!(vf, f, v)
end

@params mutable struct LinAggregation{T} <: AbstractAggregation{T}
    f::AbstractFunction{T}
    fval::AbstractVector{T}
    weights::AbstractVector{T}
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end

function LinAggregation(f::AbstractFunction{T}, weights::AbstractVector{T}; maxfevals = 10^8) where {T}
    grad = similar(f.grad)
    fval = similar(grad, TopOpt.dim(f))
    return LinAggregation(f, fval, weights, grad, 0, maxfevals)
end
# To be defined efficiently for every block constraint
function (v::LinAggregation{T})(x, grad = v.grad) where {T}
    @assert length(v.weights) == length(v.fval)
    v.fevals += 1
    v.fval .= v.f(x)
    jtvp!(grad, v.f, x, v.weights, runf=false)
    if grad !== v.grad
        v.grad .= grad
    end
    return dot(v.fval, v.weights)
end

@params mutable struct QuadAggregation{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    fval::AbstractVector{T}
    weight::T
    max::Bool
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end
function QuadAggregation(f::AbstractFunction{T}, weight::T; max=false, maxfevals=10^8) where {T}
    grad = similar(f.grad)
    fval = similar(grad, TopOpt.dim(f))
    return QuadAggregation(f, fval, weight, max, grad, 0, maxfevals)
end
function (v::QuadAggregation{T})(x, grad = v.grad) where {T}
    @assert TopOpt.dim(v.f) == length(v.fval)
    v.fevals += 1
    v.fval .= v.f(x)
    val = v.max ? max.(v.fval, 0) : v.fval
    jtvp!(grad, v.f, x, 2*v.weight .* val, runf=false)
    if grad !== v.grad
        v.grad .= grad
    end
    return v.weight * dot(val, val)
end

@params mutable struct LinQuadAggregation{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    fval::AbstractVector{T}
    lin_weights::AbstractVector{T}
    quad_weight::T
    max::Bool
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end
function LinQuadAggregation(f::AbstractFunction{T}, lin_weights::AbstractVector{T}, quad_weight::T; max=false, maxfevals=10^8) where {T}
    grad = similar(f.grad)
    fval = similar(grad, TopOpt.dim(f))
    return LinQuadAggregation(f, fval, lin_weights, quad_weight, max, grad, 0, maxfevals)
end
function (v::LinQuadAggregation{T})(x, grad = v.grad) where {T}
    @assert TopOpt.dim(v.f) == length(v.fval) == length(v.lin_weights)
    v.fevals += 1
    v.fval .= v.f(x)
    val = v.max ? max.(v.fval, 0) : v.fval
    jtvp!(grad, v.f, x, 2 * v.quad_weight .* val .+ v.lin_weights, runf=false)
    if grad !== v.grad 
        v.grad .= grad
    end
    return v.quad_weight * dot(val, val) + dot(v.fval, v.lin_weights)
end
