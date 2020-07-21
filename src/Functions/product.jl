@params mutable struct Product{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    s
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end
TopOpt.dim(p::Product) = dim(p.f)
Base.:*(f::AbstractFunction, s::Real) = s*f
function Base.:*(s, f::AbstractFunction{T}) where {T}
    return Product(f, s, similar(f.grad, T), 0, 10^8)
end

@inline function Base.getproperty(vf::Product, f::Symbol)
    f === :reuse && return vf.f.reuse
    f === :solver && return vf.f.solver
    f === :val && return (vf.s * vf.f.val)
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::Product, f::Symbol, v)
    f === :reuse && return setproperty!(vf.f, f, v)
    return setfield!(vf, f, v)
end

function (v::Product{T})(x, grad = v.grad) where {T}
    v.fevals += 1
    t = v.s * v.f(x)
    if dim(v.f) == 1
        mul!(grad, v.s, v.f.grad)
        if grad !== v.grad
            v.grad .= grad
        end
    end
    return t
end
function TopOpt.jtvp!(out, f::Product, x, v; runf=true)
    if runf
        f.f(x)
    end
    new_v = f.s * v
    jtvp!(out, f.f, x, new_v, runf=false)
    return out
end
