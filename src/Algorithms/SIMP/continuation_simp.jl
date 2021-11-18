"""
Continuous SIMP algorithm, see [TarekRay2020](@cite).
"""
@params mutable struct ContinuationSIMP <: AbstractSIMP
    simp::Any
    result::Any
    options::Any
    callback::Any
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ContinuationSIMP) =
    println("TopOpt continuation SIMP algorithm")

@params struct CSIMPOptions
    p_cont::Any
    option_cont::Any
    reuse::Bool
    steps::Integer
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::CSIMPOptions) =
    println("TopOpt continuation SIMP options")

function CSIMPOptions(
    ::Type{T} = Float64;
    steps = 40,
    p_gen = nothing,
    initial_options = MMAOptions(),
    pstart = T(1),
    pfinish = T(5),
    reuse = false,
    options_gen = nothing,
) where {T}

    if p_gen == nothing
        p_cont = PowerContinuation{T}(;
            b = T(1),
            start = pstart,
            steps = steps + 1,
            finish = pfinish,
        )
    else
        @assert steps == p_gen.length - 1
        p_cont = p_gen
    end

    if options_gen == nothing
        options_cont = MMAOptionsGen(steps = steps, initial_options = initial_options)
    else
        options_cont = options_gen
    end

    return CSIMPOptions(p_cont, options_cont, reuse, steps)
end

function ContinuationSIMP(
    simp::SIMP{T},
    steps::Int = 40,
    options::CSIMPOptions = CSIMPOptions(
        T,
        steps = steps,
        initial_options = deepcopy(simp.optimizer.options),
    ),
    callback = (i) -> (),
) where {T}
    @assert steps == options.steps
    ncells = getncells(simp.solver.problem)
    result = NewSIMPResult(T, simp.optimizer, ncells)
    return ContinuationSIMP(simp, result, options, callback)
end

# Unused functions
default_ftol(solver) = xmin / 10
function default_grtol(solver)
    @unpack xmin, problem = solver
    return xmin / 1000 * sum(problem.elementinfo.cellvolumes) * YoungsModulus(problem)
end

function MMAOptionsGen(;
    steps::Int = 40,
    initial_options = MMAOptions(),
    ftol_gen = nothing,
    xtol_gen = nothing,
    grtol_gen = nothing,
    kkttol_gen = nothing,
    maxiter_gen = nothing,
    outer_maxiter_gen = nothing,
    s_init_gen = nothing,
    s_incr_gen = nothing,
    s_decr_gen = nothing,
    store_trace_gen = nothing,
    show_trace_gen = nothing,
    auto_scale_gen = nothing,
    dual_options_gen = nothing,
)
    if maxiter_gen == nothing
        maxiter_cont = FixedContinuation(initial_options.maxiter, steps + 1)
    else
        @assert steps == maxiter_gen.length - 1
        maxiter_cont = maxiter_gen
    end

    if outer_maxiter_gen == nothing
        outer_maxiter_cont = FixedContinuation(initial_options.outer_maxiter, steps + 1)
    else
        @assert steps == outer_maxiter_gen.length - 1
        outer_maxiter_cont = outer_maxiter_gen
    end

    if xtol_gen == nothing
        xtol_cont = FixedContinuation(initial_options.tol.x, steps + 1)
    else
        @assert steps == xtol_gen.length - 1
        xtol_cont = xtol_gen
    end

    if ftol_gen == nothing
        ftol_cont = FixedContinuation(initial_options.tol.frel, steps + 1)
    else
        @assert steps == ftol_gen.length - 1
        ftol_cont = ftol_gen
    end

    if kkttol_gen == nothing
        kkttol_cont = FixedContinuation(initial_options.tol.kkt, steps + 1)
    else
        @assert steps == kkttol_gen.length - 1
        kkttol_cont = kkttol_gen
    end

    tol_cont = Nonconvex.Tolerance(x = xtol_cont, f = ftol_cont, kkt = kkttol_cont)

    if s_init_gen == nothing
        s_init_cont = FixedContinuation(initial_options.s_init, steps + 1)
    else
        @assert steps == s_init_gen.length - 1
        s_init_cont = s_init_gen
    end

    if s_incr_gen == nothing
        s_incr_cont = FixedContinuation(initial_options.s_incr, steps + 1)
    else
        @assert steps == s_incr_gen.length - 1
        s_incr_cont = s_incr_gen
    end
    if s_decr_gen == nothing
        s_decr_cont = FixedContinuation(initial_options.s_decr, steps + 1)
    else
        @assert steps == s_decr_gen.length - 1
        s_decr_cont = s_decr_gen
    end

    if store_trace_gen == nothing
        store_trace_cont = FixedContinuation(initial_options.store_trace, steps + 1)
    else
        @assert steps == store_trace_gen.length - 1
        store_trace_cont = store_trace_gen
    end

    if show_trace_gen == nothing
        show_trace_cont = FixedContinuation(initial_options.show_trace, steps + 1)
    else
        @assert steps == show_trace_gen.length - 1
        show_trace_cont = show_trace_gen
    end

    if auto_scale_gen == nothing
        auto_scale_cont = FixedContinuation(initial_options.auto_scale, steps + 1)
    else
        @assert steps == auto_scale_gen.length - 1
        auto_scale_cont = auto_scale_gen
    end

    if dual_options_gen == nothing
        dual_options_cont = FixedContinuation(initial_options.dual_options, steps + 1)
    else
        @assert steps == dual_options_gen.length - 1
        dual_options_cont = dual_options_gen
    end

    return MMAOptionsGen(
        maxiter_cont,
        outer_maxiter_cont,
        tol_cont,
        s_init_cont,
        s_incr_cont,
        s_decr_cont,
        store_trace_cont,
        show_trace_cont,
        auto_scale_cont,
        dual_options_cont,
    )
end

function (c_simp::ContinuationSIMP)(x0 = copy(c_simp.simp.solver.vars))
    @unpack optimizer = c_simp.simp
    @unpack workspace = optimizer
    @unpack options = workspace

    update!(c_simp, 1)
    # Does the first function evaluation
    Nonconvex.NonconvexCore.reset!(workspace, x0)
    c_simp.callback(0)
    r = c_simp.simp(workspace)

    f_x_previous = workspace.solution.prevf
    g_previous = copy(c_simp.simp.optimizer.workspace.solution.g)

    original_maxiter = options.maxiter
    for i = 1:c_simp.options.steps
        c_simp.callback(i)
        update!(c_simp, i + 1)
        Nonconvex.NonconvexCore.reset!(workspace)

        maxiter = 1
        workspace.options.maxiter = maxiter
        r = c_simp.simp(workspace)

        if !workspace.solution.convstate.converged
            while !workspace.solution.convstate.converged
                maxiter += 1
                workspace.options.maxiter = maxiter
                c_simp.simp(workspace)
            end
        end
        f_x_previous = workspace.solution.prevf
        g_previous = copy(c_simp.simp.optimizer.workspace.solution.g)

        workspace.options.maxiter = original_maxiter
    end
    c_simp.result = c_simp.simp.result
end

function update!(c_simp::ContinuationSIMP, i)
    p = c_simp.options.p_cont(i)
    setpenalty!(c_simp.simp, p)
    options = c_simp.options.option_cont(c_simp.simp.optimizer.workspace.options, i)
    c_simp.simp.optimizer.workspace.options = options
    return c_simp
end
frac(x) = 2 * min(abs(x), abs(x - 1))

function undo_values!(workspace, f_x_previous, g_previous)
    workspace.solution.f = workspace.solution.prevf
    workspace.solution.prevf = f_x_previous
    workspace.solution.g .= g_previous
    workspace.solution.x .= workspace.solution.prevx
    workspace.solution.prevx .= workspace.tempx
    return workspace
end
