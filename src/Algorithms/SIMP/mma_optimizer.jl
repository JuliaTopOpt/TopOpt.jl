using NonconvexMMA

abstract type AbstractOptimizer end

@params mutable struct Optimizer{Tdev} <: AbstractOptimizer
    model::AbstractModel
    alg
    workspace
    dev::Tdev
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Optimizer) = println("TopOpt optimizer")

multol!(o::Optimizer, m::Real) = o.options.tol *= m
function setbounds!(o::Optimizer, x, w)
    o.model.box_min .= max.(0, x .- w)
    o.model.box_max .= min.(1, x .+ w)
end

function Optimizer(
    obj,
    constr,
    vars,
    opt = MMA87(),
    device::Tdev = CPU();
    options = MMAOptions(),
    convcriteria = KKTCriteria(),
) where {Tdev <: AbstractDevice}

    T = eltype(vars)
    nvars = length(vars)
    x0 = copy(vars)
    model = Nonconvex.Model(obj)
    addvar!(model, zeros(T, nvars), ones(T, nvars))
    add_ineq_constraint!(model, Nonconvex.FunctionWrapper(constr, length(constr(x0))))
    @show typeof(model)
    workspace = NonconvexMMA.Workspace(model, opt, x0; options = options, convcriteria = convcriteria)
    return Optimizer(model, opt, workspace, device)
end

Utilities.getpenalty(o::Optimizer) = getpenalty(o.solver)
function Utilities.setpenalty!(o::Optimizer, p)
    setpenalty!(o.solver, p)
    return o
end

function (o::Optimizer)(x0::AbstractVector)
    @unpack workspace = o
    @unpack options, model = workspace
    reset!(workspace, x0)
    setoptions!(workspace, options)
    mma_results = Nonconvex.optimize!(workspace)
    return mma_results
end
function setoptions!(workspace::Nonconvex.Workspace, options)
    workspace.options = options
    return workspace
end

# For adaptive SIMP
function (o::Optimizer)(workspace::Nonconvex.Workspace)
    mma_results = Nonconvex.optimize!(workspace)
    return mma_results
end

@params struct MMAOptionsGen
    maxiter
    outer_maxiter
    tol
    s_init
    s_incr
    s_decr
    store_trace
    show_trace
    auto_scale
    dual_options
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::MMAOptionsGen) = println("TopOpt MMA options generator")
function (g::MMAOptionsGen)(i)
    Nonconvex.MMAOptions(
        maxiter = g.maxiter(i),
        outer_maxiter = g.outer_maxiter(i),
        tol = g.tol(i),
        s_init = g.s_init(i),
        s_incr = g.s_incr(i),
        s_decr = g.s_decr(i),
        store_trace = g.store_trace(i),
        show_trace = g.show_trace(i),
        auto_scale = g.auto_scale(i),
        dual_options = g.dual_options(i)
    )
end

function (g::MMAOptionsGen)(options, i)
    Nonconvex.MMAOptions(
        maxiter = optionalcall(g, :maxiter, options, i),
        outer_maxiter = optionalcall(g, :outer_maxiter, options, i),
        tol = optionalcall(g, :tol, options, i),
        s_init = optionalcall(g, :s_init, options, i),
        s_incr = optionalcall(g, :s_incr, options, i),
        s_decr = optionalcall(g, :s_decr, options, i),
        store_trace = optionalcall(g, :store_trace, options, i),
        show_trace = optionalcall(g, :show_trace, options, i),
        auto_scale = optionalcall(g, :auto_scale, options, i),
        dual_options = optionalcall(g, :dual_options, options, i)
    )
end
function optionalcall(g, s, options, i)
    if getproperty(g, s) isa Nothing
        return getproperty(options, s)
    else
        return getproperty(g, s)(i)
    end
end
