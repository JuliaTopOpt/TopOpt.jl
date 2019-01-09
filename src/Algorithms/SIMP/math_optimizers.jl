abstract type AbstractOptimizer end

mutable struct MMAOptimizer{T, TM <: Model{T}, TO, TSO, TObj, TConstr, TW, TState <: ConvergenceState, TOptions <: MMA.Options} <: AbstractOptimizer
    model::TM
    optimizer::TO
    suboptimizer::TSO
    obj::TObj
    constr::TConstr
    workspace::TW
    convstate::TState
    options::TOptions
end
GPUUtils.whichdevice(o::MMAOptimizer) = o.model

function Functions.maxedfevals(o::MMAOptimizer)
    maxedfevals(o.obj) || maxedfevals(o.constr)
end
@inline function Functions.maxedfevals(c::Tuple{Vararg{Constraint}})
    maxedfevals(c[1]) || maxedfevals(Base.tail(c))
end
function Functions.maxedfevals(c::Vector{<:Constraint})
    all(c -> maxedfevals(c), c)
end

MMAOptimizer(args...; kwargs...) = MMAOptimizer{CPU}(args...; kwargs...)
MMAOptimizer{T}(args...; kwargs...) where T = MMAOptimizer(T(), args...; kwargs...)
function MMAOptimizer(  device::Tdev, 
                        obj::Objective{<:AbstractFunction{T}}, 
                        constr, 
                        opt = MMA.MMA87(), 
                        subopt = Optim.ConjugateGradient(), 
                        options = MMA.Options()
                    ) where {T, Tdev}

    solver = getsolver(obj)
    nvars = length(solver.vars)
    xmin = solver.xmin

    @unpack ftol, grtol, xtol, maxiter, store_trace, extended_trace = options
    model = Model{Tdev}(nvars, obj, maxiter=maxiter, ftol=ftol, grtol=grtol, xtol=xtol, store_trace=store_trace, extended_trace=extended_trace)

    box!(model, zero(T), one(T))
    ineq_constraint!(model, constr)

    if Tdev <: CPU && whichdevice(obj) isa GPU
        x0 = Array(solver.vars)
    else
        x0 = solver.vars
    end

    @unpack s_init, s_incr, s_decr, dual_caps = options
    workspace = MMA.Workspace(model, x0, opt, subopt; s_init=s_init, 
    s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps)    

    return MMAOptimizer(model, opt, subopt, obj, constr, workspace, ConvergenceState(T), options)
end

Utilities.getpenalty(o::MMAOptimizer) = getpenalty(o.obj)
Utilities.setpenalty!(o::MMAOptimizer, p) = setpenalty!(o.obj, p)

function (o::MMAOptimizer)(x0::AbstractVector)
    @unpack workspace, options = o
    @unpack model, x, g, ∇f_x = workspace
    reset_workspace!(workspace)
    setoptions!(workspace, options)
    x .= x0
    workspace.f_x = MMA.eval_objective(model, x, ∇f_x)
    n_i = length(MMA.constraints(model))
    map!(g, 1:n_i) do i 
        @views MMA.eval_constraint(model, i, x, ∇g[:,i])
    end
    mma_results = MMA.optimize!(workspace)
    pack_results!(o, mma_results)
    return mma_results
end
function setoptions!(workspace, options)
    @unpack model = workspace
    @unpack store_trace, show_trace, extended_trace, dual_caps = options
    @unpack maxiter, ftol, xtol, grtol, subopt_options = options
    
    @pack! workspace = dual_caps, subopt_options
    model.ftol[], model.xtol[], model.grtol[] = ftol, xtol, grtol
    model.show_trace[], model.extended_trace[] = show_trace, extended_trace
    model.store_trace[] = store_trace

    return workspace
end

function reset_workspace!(workspace::Workspace{T}) where T
    @unpack primal_data, f_x = workspace
    primal_data.r0[] = 0
    primal_data.f_x[] = f_x
    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false
    f_increased, converged = false, false
    x_residual, f_residual, gr_residual = T(Inf), T(Inf), T(Inf)
    outer_iter, iter, f_calls, g_calls = 0, 0, 1, 1
    f_x_previous = T(NaN)

    @pack! workspace = x_converged, f_converged, gr_converged
    @pack! workspace = x_residual, f_residual, gr_residual
    @pack! workspace = outer_iter, iter, f_calls, g_calls
    @pack! workspace = f_x_previous
    # Maybe should remove?
    @pack! workspace = f_increased, converged

    return workspace
end

# For adaptive SIMP
function (o::MMAOptimizer)(workspace::MMA.Workspace)
    mma_results = MMA.optimize!(workspace)
    pack_results!(o, mma_results)
    return mma_results
end

function pack_results!(o, r)
    o.x_abschange = r.x_abschange
    o.x_converged = r.x_converged
    o.f_abschange = r.f_abschange
    o.f_converged = r.f_converged
    o.g_residual = r.g_residual
    o.g_converged = r.g_converged

    return o
end


@with_kw struct MMAOptionsGen{Titer, Txtol, Tftol, Tgrtol, Tsinit, Tsincr, Tsdecr, Tsuboptions}
    maxiter_cont::Titer
    xtol_cont::Txtol
    ftol_cont::Tftol
    grtol_cont::Tgrtol
    s_init_cont::Tsinit
    s_incr_cont::Tsincr
    s_decr_cont::Tsdecr
    subopt_options_cont::Tsuboptions
end
function (g::MMAOptionsGen)(i)
    MMA.Options(
        g.maxiter_cont(i),
        g.xtol_cont(i),
        g.ftol_cont(i),
        g.grtol_cont(i),
        g.s_init_cont(i),
        g.=s_incr_cont(i),
        g.s_decr_cont(i),
        g.subopt_options_cont(i)
    )
end

function (g::MMAOptionsGen)(options, i)
    MMA.Options(
        optionalcall(g.maxiter_cont, options, i),
        optionalcall(g.xtol_cont, options, i),
        optionalcall(g.ftol_cont, options, i),
        optionalcall(g.grtol_cont, options, i),
        optionalcall(g.s_init_cont, options, i),
        optionalcall(g.s_incr_cont, options, i),
        optionalcall(g.s_decr_cont, options, i),
        optionalcall(g.subopt_options_cont, options, i)
    )
end
function optionalcall(f, options, i)
    if f isa Nothing
        return options.maxiter
    else
        return f(i)
    end
end
