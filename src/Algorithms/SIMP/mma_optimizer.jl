abstract type AbstractOptimizer end

@params mutable struct MMAOptimizer{T} <: AbstractOptimizer
    model::Model{T}
    mma_alg
    suboptimizer
    obj
    constr
    workspace
    convstate::ConvergenceState
    options::MMA.Options
end
multol!(o::MMAOptimizer, m::Real) = o.options.tol *= m
function setbounds!(o::MMAOptimizer, x, w)
    o.model.box_min .= max.(0, x .- w)
    o.model.box_max .= min.(1, x .+ w)
end

function Functions.maxedfevals(o::MMAOptimizer)
    maxedfevals(o.obj) || maxedfevals(o.constr)
end
@inline function Functions.maxedfevals(c::Tuple{Vararg{Constraint}})
    maxedfevals(c[1]) || maxedfevals(Base.tail(c))
end
function Functions.maxedfevals(c::Vector{<:Constraint})
    all(c -> maxedfevals(c), c)
end

function MMAOptimizer(args...; kwargs...)
    return MMAOptimizer{CPU}(args...; kwargs...)
end
function MMAOptimizer{T}(args...; kwargs...) where T
    return MMAOptimizer(T(), args...; kwargs...)
end
function MMAOptimizer{T}(::AbstractDevice, args...; kwargs...) where T
    throw("Check your types.")
end
function MMAOptimizer(  device::Tdev, 
                        obj::Objective{T, <:AbstractFunction{T}}, 
                        constr, 
                        opt = MMA.MMA87(), 
                        subopt = Optim.ConjugateGradient();
                        options = MMA.Options(),
                        convcriteria = MMA.KKTCriteria()
                    ) where {T, Tdev <: AbstractDevice}

    solver = getsolver(obj)
    nvars = length(solver.vars)
    xmin = solver.xmin

    model = Model{Tdev}(nvars, obj)

    box!(model, zero(T), one(T))
    ineq_constraint!.(Ref(model), constr)

    if Tdev <: CPU && whichdevice(obj) isa GPU
        x0 = Array(solver.vars)
    else
        x0 = solver.vars
    end

    workspace = MMA.Workspace(model, x0, opt, subopt; options = options, convcriteria = convcriteria)

    return MMAOptimizer(model, opt, subopt, obj, constr, workspace, ConvergenceState(T), options)
end

Utilities.getpenalty(o::MMAOptimizer) = getpenalty(o.obj)
function Utilities.setpenalty!(o::MMAOptimizer, p)
    setpenalty!(o.obj, p)
    setpenalty!.(o.constr, p)
end

function (o::MMAOptimizer)(x0::AbstractVector)
    @unpack workspace, options = o
    @unpack model = workspace
    @unpack x, g, ∇g, ∇f_x = workspace.primal_data

    reset!(workspace, x0)
    setoptions!(workspace, options)
    mma_results = MMA.optimize!(workspace)
    o.convstate, mma_results.convstate
    return mma_results
end
function setoptions!(workspace, options)
    @unpack options = workspace
    @unpack store_trace, show_trace, extended_trace, dual_caps = options
    @unpack maxiter, tol, subopt_options = options
    @pack! options = dual_caps, subopt_options, tol
    @pack! options = show_trace, extended_trace, store_trace

    return workspace
end

function reset!(w::Workspace, args...)
    @unpack primal_data, model = w
    # primal_data.r0 = 0
    outer_iter, iter, f_calls, g_calls = 0, 0, 0, 0
    @pack! w = f_calls, g_calls
    MMA.update_values!(w, args...)
    # Assess multiple types of convergence
    w.convstate = MMA.assess_convergence(w)
    @pack! w = outer_iter, iter
    return w
end

# For adaptive SIMP
function (o::MMAOptimizer)(workspace::MMA.Workspace)
    mma_results = MMA.optimize!(workspace)
    return mma_results
end

@params struct MMAOptionsGen
    maxiter
    outer_maxiter
    tol
    s_init
    s_incr
    s_decr
    dual_caps
    store_trace
    show_trace
    extended_trace
    subopt_options
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
