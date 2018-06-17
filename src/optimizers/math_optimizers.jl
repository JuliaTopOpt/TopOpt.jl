abstract type AbstractOptimizer end

mutable struct MMAOptimizer{T,TI,TO,TObj,TConstr} <: AbstractOptimizer
    model::MMAModel{T,TI}
    s_init::T
    s_decr::T
    s_incr::T
    optimizer::Type{TO}
    obj::TObj
    constr::TConstr
    x_abschange::T
    x_converged::Bool
    f_abschange::T
    f_converged::Bool
    g_residual::T
    g_converged::Bool
end

function MMAOptimizer(obj, constr, ::Type{TO}=Optim.AcceleratedGradientDescent;
    max_iters = 100,
    x_tol = 0.001,
    f_tol = x_tol/2,
    gr_tol = f_tol/10,
    s_init = 0.5,
    s_incr = 1.05,
    s_decr = 0.65) where {TO}

    nvars = length(obj.solver.vars)
    xmin = obj.solver.xmin
    T = typeof(xmin)
    model = MMAModel{T,Int}(nvars, obj, max_iters=max_iters, ftol=f_tol, grtol=gr_tol, xtol=x_tol, store_trace=false, extended_trace=false)

    box!(model, zero(T), one(T))
    ineq_constraint!(model, constr)

    return MMAOptimizer{T,Int,TO,typeof(obj),typeof(constr)}(model, s_init, s_decr, s_incr, TO, obj, constr, T(NaN), false, T(NaN), false, T(NaN), false)
end

function MMAOptimizer(to, obj, constr, ::Type{TO}=Optim.ConjugateGradient;
    max_iters = 100,
    x_tol = 0.001,
    f_tol = x_tol/2,
    gr_tol = f_tol/10,
    s_init = 0.5,
    s_incr = 1.05,
    s_decr = 0.65) where {TO}

    nvars = length(obj.solver.vars)
    xmin = obj.solver.xmin
    T = typeof(xmin)
    model = MMAModel{T,Int}(nvars, (x, grad) -> obj(to, x, grad), max_iters=max_iters, ftol=f_tol, grtol=gr_tol, xtol=x_tol, store_trace=false, extended_trace=false)

    box!(model, zero(T), one(T))
    ineq_constraint!(model, constr)

    return MMAOptimizer{T,Int,TO,typeof(obj),typeof(constr)}(model, s_init, s_decr, s_incr, TO, obj, constr, T(NaN), false, T(NaN), false, T(NaN), false)
end

function (o::MMAOptimizer)(to, x0)
    mma_results = optimize(to, o.model, x0, o.optimizer, o.s_init, o.s_incr, o.s_decr)
    #@code_warntype optimize(to, o.model, o.obj.solver.vars, o.optimizer, o.s_init, o.s_incr, o.s_decr)

    o.x_abschange = mma_results.x_abschange
    o.x_converged = mma_results.x_converged
    o.f_abschange = mma_results.f_abschange
    o.f_converged = mma_results.f_converged
    o.g_residual = mma_results.g_residual
    o.g_converged = mma_results.g_converged

    return mma_results
end

function (o::MMAOptimizer)(x0)
    mma_results = optimize(o.model, x0, o.optimizer, o.s_init, o.s_incr, o.s_decr)
    #@code_warntype optimize(o.model, o.obj.solver.vars, o.optimizer, o.s_init, o.s_incr, o.s_decr)

    o.x_abschange = mma_results.x_abschange
    o.x_converged = mma_results.x_converged
    o.f_abschange = mma_results.f_abschange
    o.f_converged = mma_results.f_converged
    o.g_residual = mma_results.g_residual
    o.g_converged = mma_results.g_converged

    return mma_results
end
