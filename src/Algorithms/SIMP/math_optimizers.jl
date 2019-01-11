abstract type AbstractOptimizer end

mutable struct MMAOptimizer{T, TM <: Model{T}, TO, TSO, TObj, TConstr, TW, TState <: ConvergenceState, TOptions <: MMA.Options} <: AbstractOptimizer
    model::TM
    mma_alg::TO
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

function MMAOptimizer(args...; options = MMA.Options())
    return MMAOptimizer{CPU}(args...; options = options)
end
function MMAOptimizer{T}(args...; options = MMA.Options()) where T
    return MMAOptimizer(T(), args...; options = options)
end
function MMAOptimizer(  device::Tdev, 
                        obj::Objective{<:AbstractFunction{T}}, 
                        constr, 
                        opt = MMA.MMA87(), 
                        subopt = Optim.ConjugateGradient();
                        options = MMA.Options()
                    ) where {T, Tdev}

    solver = getsolver(obj)
    nvars = length(solver.vars)
    xmin = solver.xmin

    model = Model{Tdev}(nvars, obj)

    box!(model, zero(T), one(T))
    ineq_constraint!(model, constr)

    if Tdev <: CPU && whichdevice(obj) isa GPU
        x0 = Array(solver.vars)
    else
        x0 = solver.vars
    end

    workspace = MMA.Workspace(model, x0, opt, subopt; options = options)

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
    @unpack options = workspace
    @unpack store_trace, show_trace, extended_trace, dual_caps = options
    @unpack maxiter, ftol, xtol, grtol, subopt_options = options
    
    @pack! options = dual_caps, subopt_options
    options.ftol, options.xtol, options.grtol = ftol, xtol, grtol
    options.show_trace, options.extended_trace = show_trace, extended_trace
    options.store_trace = store_trace

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
    workspace.convstate = mma_results.convstate
    return mma_results
end

struct MMAOptionsGen{Titer, Touteriter, Ttol, Tsinit, Tsincr, Tsdecr, Tdualcaps, Tstore_trace, Tshow_trace, Text_trace, Tsuboptions}
    maxiter::Titer
    outer_maxiter::Touteriter
    tol::Ttol
    s_init::Tsinit
    s_incr::Tsincr
    s_decr::Tsdecr
    dual_caps::Tdualcaps
    store_trace::Tstore_trace
    show_trace::Tshow_trace
    extended_trace::Text_trace
    subopt_options::Tsuboptions
end
function (g::MMAOptionsGen)(i)
    MMA.Options(
        g.maxiter(i),
        g.outer_maxiter(i),
        g.tol(i),
        g.s_init(i),
        g.s_incr(i),
        g.s_decr(i),
        g.dual_caps(i),
        g.store_trace(i),
        g.show_trace(i),
        g.extended_trace(i),
        g.subopt_options(i)
    )
end

function (g::MMAOptionsGen)(options, i)
    MMA.Options(
        optionalcall(g, :maxiter, options, i),
        optionalcall(g, :outer_maxiter, options, i),
        optionalcall(g, :tol, options, i),
        optionalcall(g, :s_init, options, i),
        optionalcall(g, :s_incr, options, i),
        optionalcall(g, :s_decr, options, i),
        optionalcall(g, :dual_caps, options, i),
        optionalcall(g, :store_trace, options, i),
        optionalcall(g, :show_trace, options, i),        
        optionalcall(g, :extended_trace, options, i),
        optionalcall(g, :subopt_options, options, i)
    )
end
function optionalcall(g, s, options, i)
    if getproperty(g, s) isa Nothing
        return getproperty(options, s)
    else
        return getproperty(g, s)(i)
    end
end
