#=
mutable struct ContinuationSIMP{T, TO, TP, TMC, TPC, TXC, TFC, TGC, TInitC, TIncrC, TDecrC} <: AbstractSIMP
    simp::SIMP{T,TO,TP}
    reuse::Bool
    result::SIMPResult{T}
    steps::Int
    maxiter_cont::TMC
    maxiter::Int
    p_cont::TPC
    xtol_cont::TXC
    ftol_cont::TFC
    grtol_cont::TGC
    s_init_cont::TInitC
    s_incr_cont::TIncrC
    s_decr_cont::TDecrC
    double_solve::Bool
end
=#
mutable struct ContinuationSIMP{T, TO, TP, TPC, TFC} <: AbstractSIMP
    simp::SIMP{T,TO,TP}
    reuse::Bool
    result::SIMPResult{T}
    steps::Int
    maxiter::Int
    p_cont::TPC
    ftol_cont::TFC
end
GPUUtils.whichdevice(c::ContinuationSIMP) = whichdevice(c.simp)

function ContinuationSIMP(simp::SIMP{T}; 
    steps = 40, 
    start = T(1), 
    finish = T(5),
    reuse = true,
    p_cont = PowerContinuation{T}(b=T(1), 
        start=start,
        steps=steps+1,
        finish=finish),
    maxiter = 500,
    ftol_cont = PowerContinuation{T}(b=T(1),
        start=simp.optimizer.obj.solver.xmin/10,
        steps=steps+1,
        finish=simp.optimizer.obj.solver.xmin/10
        ),
    #=
    maxiter_cont = PowerContinuation{Int}(b=1, 
        start=simp.optimizer.model.maxiter[], 
        steps=steps+1,
        finish=simp.optimizer.model.maxiter[]),
    xtol_cont = PowerContinuation{T}(b=T(1),
        start=simp.optimizer.obj.solver.xmin,
        steps=steps+1,
        finish=simp.optimizer.obj.solver.xmin
        ),
    grtol_cont = PowerContinuation{T}(b=T(1),
        start=simp.optimizer.obj.solver.xmin/1000*simp.optimizer.constr.total_volume*YoungsModulus(simp.optimizer.obj.problem),
        steps=steps+1,
        finish=simp.optimizer.obj.solver.xmin/1000*simp.optimizer.constr.total_volume*YoungsModulus(simp.optimizer.obj.problem)
        ),
    s_init_cont = PowerContinuation{T}(b=T(1),
        start=T(0.5),
        steps=steps+1,
        finish=T(0.5)
        ),
    s_incr_cont = PowerContinuation{T}(b=T(1),
        start=T(1.2),
        steps=steps+1,
        finish=T(1.2)
        ),
    s_decr_cont = PowerContinuation{T}(b=T(1),
        start=T(0.7),
        steps=steps+1,
        finish=T(0.7)
        ),
    double_solve = false,=#
    ) where T

    ncells = getncells(simp.optimizer.obj.f.problem)
    result = NewSIMPResult(T, simp.optimizer, ncells)

    return ContinuationSIMP(simp, reuse, result, steps, maxiter, p_cont, ftol_cont)

    # return ContinuationSIMP(simp, reuse, result, steps, maxiter_cont, maxiter, p_cont, xtol_cont, ftol_cont, grtol_cont, s_init_cont, s_incr_cont, s_decr_cont, double_solve)
end

function (c_simp::ContinuationSIMP{<:Any,<:MMAOptimizer})(x0=c_simp.simp.optimizer.obj.solver.vars, terminate_early=false)
    @unpack model, s_init, s_decr, s_incr, optimizer, suboptimizer, dual_caps, obj, constr, x_abschange, x_converged, f_abschange, f_converged, g_residual, g_converged = c_simp.simp.optimizer

    c_simp.simp.optimizer.model.xtol[] = 0
    c_simp.simp.optimizer.model.grtol[] = 0
    update!(c_simp, 1)

    # Does the first function evaluation
    # Number of function evaluations is the number of iterations plus 1
    setreuse!(c_simp.simp.optimizer, false)

    workspace = MMA.MMAWorkspace(model, x0, optimizer, suboptimizer, s_init=s_init, s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps)

    if maxedfevals(c_simp.simp.optimizer)
        c_simp.result = c_simp.simp.result
        return c_simp.result
    end
    c_simp.simp(workspace)

    maxiter = workspace.model.maxiter[]
    #fevals_hist = zeros(Int, 3)
    #fevals_hist .= getfevals(c_simp.simp.optimizer)
    for i in 1:c_simp.steps
        maxedfevals(c_simp.simp.optimizer) && break
        workspace.iter = workspace.outer_iter = 0
        workspace.model.maxiter[] = 1

        workspace.converged = false
        if i == c_simp.steps
            setreuse!(c_simp.simp.optimizer, false)
        else
            setreuse!(c_simp.simp.optimizer, c_simp.reuse)
        end
        ftol = update!(c_simp, i+1)

        c_simp.simp(workspace)
        if getreuse(c_simp.simp.optimizer)
            c_simp.simp.optimizer.workspace.f_x = c_simp.simp.optimizer.workspace.f_x_previous
            c_simp.simp.optimizer.workspace.f_x_previous = f_x_previous
            #c_simp.simp.optimizer.workspace.g .= g_previous
            c_simp.simp.optimizer.workspace.x .= c_simp.simp.optimizer.workspace.x1
            c_simp.simp.optimizer.workspace.x1 .= c_simp.simp.optimizer.workspace.x2
        end
    
        if !c_simp.simp.optimizer.workspace.converged
            setreuse!(c_simp.simp.optimizer, false)
            while !workspace.converged && !maxedfevals(c_simp.simp.optimizer)
                workspace.model.maxiter[] += 1
                c_simp.simp(workspace)
            end
        end
        f_x_previous = c_simp.simp.optimizer.workspace.f_x_previous
        g_previous = copy(c_simp.simp.optimizer.workspace.g)
        #for j in length(fevals_hist)-1:-1:2
        #    fevals_hist[j] = fevals_hist[j-1]
        #end

        maxedfevals(c_simp.simp.optimizer) && break
        #=
        if terminate_early
            x_hist = obj.topopt_trace.x_hist
            x = x_hist[fevals]
            same = true
            for ind in 2:length(fevals_hist)
                x1 = x_hist[fevals_hist[ind]]
                for j in 1:length(x)
                    if round(x[j]) != round(x1[j])
                        same = false
                        break
                    end
                end
            end
            fractionness = sum(frac, x) / length(x)
            same && fractionness <= 0.01 && break
        end
        =#
    end
    workspace.model.maxiter[] = maxiter
    c_simp.result = c_simp.simp.result
end

function update!(c_simp::ContinuationSIMP{<:Any,<:MMAOptimizer}, i)
    p = c_simp.p_cont(i)
    setpenalty!(c_simp.simp, p)
    ftol = c_simp.ftol_cont(i)
    c_simp.simp.optimizer.model.ftol[] = ftol

    return ftol
end
frac(x) = 2*min(abs(x), abs(x - 1))
