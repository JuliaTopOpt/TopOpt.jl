@params mutable struct BoxOptimizer{T} <: AbstractOptimizer
    obj
    lb
    ub
    suboptimizer
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
                        obj::Objective{T, <:AbstractFunction{T}}, 
                        subopt = Optim.ConjugateGradient();
                        options = Optim.Options(allow_f_increases=true, x_tol=-1e-4, f_tol=-1e-4)
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
    @unpack options, obj, lb, ub, subopt = o
    function fg!(F,G,x)
        if G != nothing && F != nothing
            return obj(x, G)
        elseif F != nothing
            return obj(x)
        else
            return nothing
        end
    end
    return Optim.optimize(Optim.only_fg!(fg!), lb, ub, x0, Optim.Fminbox(subopt), options)
end
