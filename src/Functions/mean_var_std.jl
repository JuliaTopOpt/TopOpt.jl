@params struct MeanVar{T, Tblock <: AbstractFunction{T}} <: AbstractFunction{T}
    block::Tblock
    w::T
    grad::AbstractVector{T}
end
Nonconvex.getdim(::MeanVar) = 1
MeanVar(block::AbstractFunction{T}, w::T) where {T} = MeanVar(block, w, zeros(T, TopOpt.getnvars(block)))
@forward_property MeanVar block

function (f::MeanVar)(x, grad = f.grad)
    @unpack block, w = f
    val = block(x)
    N = length(val)
    m = mean(val)
    v = var(val)
    out = m + w*v
    grad .= 0
    v = 2*w/(N-1) .* (val .- m) .+ 1/N
    TopOpt.jtvp!(grad, block, x, v; runf=false)
    if grad !== f.grad
        f.grad .= grad
    end
    return out
end

@params struct MeanStd{T, Tblock <: AbstractFunction{T}} <: AbstractFunction{T}
    block::Tblock
    w::T
    grad::AbstractVector{T}
end
Nonconvex.getdim(::MeanStd) = 1
MeanStd(block::AbstractFunction{T}, w::T) where {T} = MeanStd(block, w, zeros(T, TopOpt.getnvars(block)))
@forward_property MeanStd block

function (f::MeanStd)(x, grad = f.grad)
    @unpack block, w = f
    val = block(x)
    N = length(val)
    m = mean(val)
    s = std(val)
    out = m + w*s
    grad .= 0
    v = 1/N .+ w/(s*(N-1)) .* (val .- m)
    TopOpt.jtvp!(grad, block, x, v; runf=false)
    if grad !== f.grad
        f.grad .= grad
    end
    return out
end

@params struct ScalarValued{T, Tblock <: AbstractFunction{T}} <: AbstractFunction{T}
    block::Tblock
    f
    grad::AbstractVector{T}
end
Nonconvex.getdim(::ScalarValued) = 1
ScalarValued(block::AbstractFunction{T}, f) where {T} = ScalarValued(block, f, zeros(T, TopOpt.getnvars(block)))
@forward_property ScalarValued block

function (func::ScalarValued{T})(x, grad = func.grad) where {T}
    @unpack block, f = func
    val = block(x)
    N = length(val)
    out_tracked, ȳ = Tracker.forward(f, val)
    out::T = Tracker.data(out_tracked)
    ∂out∂val::typeof(val) = Tracker.data(ȳ(1)[1])
    grad .= 0
    v = ∂out∂val
    TopOpt.jtvp!(grad, block, x, v; runf=false)
    if grad !== func.grad
        func.grad .= grad
    end
    return out
end

#=
function (func::ScalarValued{T})(x, grad = func.grad) where {T}
    @unpack block, f = func
    val = block(x)
    N = length(val)
    out_tracked, ȳ = Tracker.forward(f, val)
    out::T = Tracker.data(out_tracked)
    ∂out∂val::typeof(val) = Tracker.data(ȳ(1)[1])
    new_grad = similar(grad)
    v = ∂out∂val
    TopOpt.jtvp!(new_grad, block, x, v; runf=false)
    grad .= 0.9 .* func.grad .+ 0.1 .* new_grad
    if grad !== func.grad
        func.grad .= grad
    end
    return out
end
=#
