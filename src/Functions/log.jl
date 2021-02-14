@params mutable struct Log{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end
function Base.log(f::AbstractFunction{T}) where {T}
    return Log(f, similar(f.grad, T), 0, 10^8)
end

Nonconvex.getdim(l::Log) = Nonconvex.getdim(l.f)
@inline function Base.getproperty(vf::Log, f::Symbol)
    f === :f && return getfield(vf, :f)
    f === :grad && return getfield(vf, :grad)
    f === :fevals && return getfield(vf, :fevals)
    f === :maxfevals && return getfield(vf, :maxfevals)
    return getproperty(getfield(vf, :f), f)
end
@inline function Base.setproperty!(vf::Log, f::Symbol, v)
    f === :f && return setfield!(vf, :f, v)
    f === :grad && return setfield!(vf, :grad, v)
    f === :fevals && return setfield!(vf, :fevals, v)
    f === :maxfevals && return setfield!(vf, :maxfevals, v)
    return setproperty!(getfield(vf, :f), f, v)
end

function (v::Log{T})(x, grad = v.grad) where {T}
    v.fevals += 1
    t = v.f(x) .+ sqrt(eps(T))
    if Nonconvex.getdim(v.f) == 1
        grad .= v.f.grad ./ t
        if grad !== v.grad
            v.grad .= grad
        end
    end
    return log.(t)
end
function TopOpt.jtvp!(out, f::Log, x, v; runf=true)
    if runf
        t = f.f(x) .+ sqrt(eps(T))
    else
        t = f.f.val
    end
    new_v = v ./ t
    jtvp!(out, f.f, x, new_v, runf=false)
    return out
end
