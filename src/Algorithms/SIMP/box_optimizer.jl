@params mutable struct BoxOptimizer{T} <: AbstractOptimizer
    obj::AbstractFunction{T}
    lb::AbstractVector{T}
    ub::AbstractVector{T}
    suboptimizer::Optim.FirstOrderOptimizer
    options::Optim.Options
end
function setbounds!(o::BoxOptimizer, x, w)
    o.lb .= max.(0, x .- w)
    o.ub .= min.(1, x .+ w)
end

Functions.maxedfevals(o::BoxOptimizer) = maxedfevals(o.obj)

function BoxOptimizer(args...; kwargs...)
    return BoxOptimizer{CPU}(args...; kwargs...)
end
function BoxOptimizer{T}(args...; kwargs...) where T
    return BoxOptimizer(T(), args...; kwargs...)
end
function BoxOptimizer{T}(::AbstractDevice, args...; kwargs...) where T
    throw("Check your types.")
end
function BoxOptimizer(  device::Tdev, 
                        obj::AbstractFunction{T}, 
                        subopt::Optim.FirstOrderOptimizer = ConjugateGradient();
                        options = Optim.Options(allow_f_increases=false, x_tol=1e-5, f_tol=1e-5, g_tol=1e-2)
                    ) where {T, Tdev <: AbstractDevice}

    solver = getsolver(obj)
    nvars = length(solver.vars)
    xmin = solver.xmin

    if Tdev <: CPU && whichdevice(obj) isa GPU
        x0 = Array(solver.vars)
    else
        x0 = solver.vars
    end
    lb = similar(x0); lb .= 0;
    ub = similar(x0); ub .= 1;

    return BoxOptimizer(obj, lb, ub, subopt, options)
end

Utilities.getpenalty(o::BoxOptimizer) = getpenalty(o.obj)
function Utilities.setpenalty!(o::BoxOptimizer, p)
    setpenalty!(o.obj, p)
end

function (o::BoxOptimizer)(x0::AbstractVector)
    @unpack options, obj, lb, ub, suboptimizer = o
    grad_func = (g, x) -> obj(x, g)
    return Optim.optimize(obj, grad_func, lb, ub, x0, Optim.Fminbox(suboptimizer), options)
end
