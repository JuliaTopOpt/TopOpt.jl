@params mutable struct Sum{T} <: AbstractFunction{T}
    f1::AbstractFunction{T}
    f2::AbstractFunction{T}
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end

function Base.:+(f1::AbstractFunction{T1}, f2::AbstractFunction{T2}) where {T1, T2}
    T = promote_type(T1, T2)
    return Sum(f1, f2, similar(f1.grad, T), 0, 10^8)
end

@inline function Base.getproperty(vf::Sum, f::Symbol)
    f === :reuse && return vf.f1.reuse && vf.f2.reuse
    f === :solver && return vf.f1.solver
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::Sum, f::Symbol, v)
    f === :reuse && return (setproperty!(vf.f1, f, v); setproperty!(vf.f2, f, v))
    return setfield!(vf, f, v)
end

function (v::Sum{T})(x, grad = v.grad) where {T}
    v.fevals += 1
    t = v.f1(x) + v.f2(x)
    grad .= v.f1.grad .+ v.f2.grad
    if grad !== v.grad
        v.grad .= grad
    end
    return t
end
