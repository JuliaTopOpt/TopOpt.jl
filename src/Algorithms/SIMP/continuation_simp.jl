@params mutable struct ContinuationSIMP <: AbstractSIMP
    simp
    result
    options
end
GPUUtils.whichdevice(c::ContinuationSIMP) = whichdevice(c.simp)

@params struct CSIMPOptions
    p_cont
    option_cont
    reuse::Bool
    steps::Integer
end

function CSIMPOptions(::Type{T} = Float64; 
                        steps = 40, 
                        p_gen = nothing,
                        initial_options = MMA.Options(), 
                        pstart = T(1), 
                        pfinish = T(5), 
                        reuse = true,
                        options_gen = nothing
                    ) where {T}

    if p_gen == nothing
        p_cont = PowerContinuation{T}( ; b = T(1), 
                                        start = pstart,
                                        steps = steps + 1,
                                        finish = pfinish
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

function ContinuationSIMP(  simp::SIMP{T},
                            steps::Int = 40, 
                            options::CSIMPOptions = CSIMPOptions(T, steps = steps, 
                                                                initial_options = deepcopy(simp.optimizer.options)
                                                    )
                        ) where T

    @assert steps == options.steps
    ncells = getncells(simp.optimizer.obj.f.problem)
    result = NewSIMPResult(T, simp.optimizer, ncells)
    return ContinuationSIMP(simp, result, options)
end

# Unused functions
default_ftol(solver) = xmin / 10
function default_grtol(solver)
    @unpack xmin, problem = solver
    return xmin / 1000 * sum(problem.elementinfo.cellvolumes) * YoungsModulus(problem)
end

function MMAOptionsGen(;    steps::Int = 40, 
                            initial_options = MMA.Options(), 
                            ftol_gen = nothing, 
                            xtol_gen = nothing,
                            grtol_gen = nothing,
                            kkttol_gen = nothing,
                            maxiter_gen = nothing,
                            outer_maxiter_gen = nothing,
                            s_init_gen = nothing,
                            s_incr_gen = nothing,
                            s_decr_gen = nothing,
                            dual_caps_gen = nothing,
                            store_trace_gen = nothing,
                            show_trace_gen = nothing,
                            extended_trace_gen = nothing,
                            subopt_options_gen = nothing
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
        xtol_cont = FixedContinuation(initial_options.xtol, steps + 1)
    else
        @assert steps == xtol_gen.length - 1
        xtol_cont = xtol_gen
    end

    if ftol_gen == nothing
        ftol_cont = FixedContinuation(initial_options.ftol, steps + 1)
    else
        @assert steps == ftol_gen.length - 1
        ftol_cont = ftol_gen
    end

    if grtol_gen == nothing
        grtol_cont = FixedContinuation(initial_options.grtol, steps + 1)
    else
        @assert steps == grtol_gen.length - 1
        grtol_cont = grtol_gen
    end

    if kkttol_gen == nothing
        kkttol_cont = FixedContinuation(initial_options.kkttol, steps + 1)
    else
        @assert steps == kkttol_gen.length - 1
        kkttol_cont = kkttol_gen
    end

    tol_cont = MMA.Tolerances(xtol_cont, ftol_cont, grtol_cont, kkttol_cont)

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

    if dual_caps_gen == nothing
        dual_caps_cont = FixedContinuation(initial_options.dual_caps, steps + 1)
    else
        @assert steps == dual_caps_gen.length - 1
        dual_caps_cont = dual_caps_gen
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

    if extended_trace_gen == nothing
        extended_trace_cont = FixedContinuation(initial_options.extended_trace, steps + 1)
    else
        @assert steps == extended_trace_gen.length - 1
        extended_trace_cont = extended_trace_gen
    end

    if subopt_options_gen == nothing
        subopt_options_cont = FixedContinuation(initial_options.subopt_options, steps + 1)
    else
        @assert steps == subopt_options_gen.length - 1
        subopt_options_cont = subopt_options_gen
    end

    return MMAOptionsGen(   maxiter_cont, 
                            outer_maxiter_cont,
                            tol_cont, 
                            s_init_cont, 
                            s_incr_cont,
                            s_decr_cont,
                            dual_caps_cont,
                            store_trace_cont,
                            show_trace_cont,
                            extended_trace_cont,
                            subopt_options_cont
                        )
end

function (c_simp::ContinuationSIMP)(x0 = copy(c_simp.simp.optimizer.obj.solver.vars), terminate_early = false)
    @unpack optimizer = c_simp.simp
    @unpack workspace, mma_alg, suboptimizer = optimizer 
    @unpack obj, constr, convstate, options = optimizer

    prev_fevals = getfevals(optimizer)
    setreuse!(optimizer, false)
    update!(c_simp, 1)
    # Does the first function evaluation
    reset!(workspace, x0)
    r = c_simp.simp(workspace, prev_fevals)

    if maxedfevals(c_simp.simp.optimizer)
        c_simp.result = c_simp.simp.result
        return c_simp.result
    end

    f_x_previous = workspace.primal_data.f_x_previous
    g_previous = copy(c_simp.simp.optimizer.workspace.primal_data.g)

    original_maxiter = options.maxiter
    for i in 1:c_simp.options.steps
        maxedfevals(optimizer) && break
        prev_fevals = getfevals(optimizer)
        reuse = i == c_simp.options.steps ? false : c_simp.options.reuse
        setreuse!(optimizer, reuse)
        update!(c_simp, i+1)
        reset!(workspace)

        maxiter = 1
        workspace.options.maxiter = maxiter
        r = c_simp.simp(workspace, prev_fevals)

        if any(getreuse(optimizer))
            undo_values!(workspace, f_x_previous, g_previous)
        end
        if !workspace.convstate.converged
            setreuse!(c_simp.simp.optimizer, false)
            while !workspace.convstate.converged && !maxedfevals(optimizer)
                maxiter += 1
                workspace.options.maxiter = maxiter
                c_simp.simp(workspace)
            end
        end
        f_x_previous = workspace.primal_data.f_x_previous
        g_previous = copy(c_simp.simp.optimizer.workspace.primal_data.g)

        workspace.options.maxiter = original_maxiter
    end
    c_simp.result = c_simp.simp.result
end

function update!(c_simp::ContinuationSIMP, i)
    p = c_simp.options.p_cont(i)
    #@show p
    setpenalty!(c_simp.simp, p)
    options = c_simp.options.option_cont(c_simp.simp.optimizer.options, i)
    c_simp.simp.optimizer.workspace.options = options

    return c_simp
end
frac(x) = 2*min(abs(x), abs(x - 1))

function undo_values!(workspace, f_x_previous, g_previous)
    workspace.primal_data.f_x = workspace.primal_data.f_x_previous
    workspace.primal_data.f_x_previous = f_x_previous
    workspace.primal_data.g .= g_previous
    #workspace.primal_data.x .= workspace.primal_data.x1
    #workspace.primal_data.x1 .= workspace.primal_data.x2
    return workspace
end
