@params mutable struct Product{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    s::T
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end

function Base.:*(f::AbstractFunction{T1}, s::T2) where {T1, T2}
    T = promote_type(T1, T2)
    return Product{T}(f, s, similar(f.grad, T), 0, 10^8)
end

@inline function Base.getproperty(vf::Product, f::Symbol)
    f === :reuse && return vf.f.reuse
    f === :solver && return vf.f.solver
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::Product, f::Symbol, v)
    f === :reuse && return setproperty!(vf.f, f, v)
    return setfield!(vf, f, v)
end

function (v::Product{T})(x, grad = v.grad) where {T}
    v.fevals += 1
    t = v.s * v.f(x)
    grad .= v.s .* v.f.grad
    if grad !== v.grad
        v.grad .= grad
    end
    return t
end
