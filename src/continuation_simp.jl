mutable struct ContinuationSIMP{T,TO,TP,TMC,TPC,TXC,TInitC,TIncrC,TDecrC}
    simp::SIMP{T,TO,TP}
    reuse::Bool
    result::SIMPResult{T}
    topologies::Vector{Vector{T}}
    steps::Int
    max_iters_cont::TMC
    p_cont::TPC
    x_tol_cont::TXC
    s_init_cont::TInitC
    s_incr_cont::TIncrC
    s_decr_cont::TDecrC
end
function ContinuationSIMP(simp::SIMP{T}; 
    steps = 40, 
    start = T(1), 
    finish = T(5),
    reuse = true,
    p_cont = PowerContinuation{T}(b=T(1), 
        start=start,
        steps=steps,
        finish=finish),
    max_iters_cont = PowerContinuation{Int}(b=1, 
        start=simp.optimizer.model.max_iters, 
        steps=steps,
        finish=simp.optimizer.model.max_iters),
    x_tol_cont = PowerContinuation{T}(b=T(1),
        start=simp.optimizer.obj.solver.xmin,
        steps=steps,
        finish=simp.optimizer.obj.solver.xmin
        ),
    s_init_cont = PowerContinuation{T}(b=T(1),
        start=T(0.5),
        steps=steps,
        finish=T(0.5)
        ),
    s_incr_cont = PowerContinuation{T}(b=T(1),
        start=T(1.05),
        steps=steps,
        finish=T(1.05)
        ),
    s_decr_cont = PowerContinuation{T}(b=T(1),
        start=T(0.65),
        steps=steps,
        finish=T(0.65)
        ),
    ) where T

    topology = fill(T(NaN), getncells(simp.optimizer.obj.problem.ch.dh.grid))
    objval = T(NaN)
    result = SIMPResult(topology, objval, T(NaN), false, T(NaN), false, T(NaN), false)

    return ContinuationSIMP(simp, reuse, result, simp.topologies, steps, max_iters_cont, p_cont, x_tol_cont, s_init_cont, s_incr_cont, s_decr_cont)
end

function (c_simp::ContinuationSIMP{<:Any,<:MMAOptimizer})(x0=c_simp.simp.optimizer.obj.solver.vars)
    update!(c_simp, 1)
    c_simp.simp(x0)
    for i in 2:c_simp.steps
        update!(c_simp, i)
        c_simp.simp()
        c_simp.simp.optimizer.obj.reuse = c_simp.reuse
    end
    c_simp.result = c_simp.simp.result
end

function update!(c_simp::ContinuationSIMP{<:Any,<:MMAOptimizer}, i)
    c = 100
    m = 100
    p = c_simp.p_cont(i)
    x_tol = c_simp.x_tol_cont(i)
    vol = c_simp.simp.optimizer.constr.total_volume
    E = YoungsModulus(c_simp.simp.optimizer.obj.problem)
    f_tol = x_tol/100
    gr_tol = f_tol/(c*m^(p-1))*vol*E
    max_iters = c_simp.max_iters_cont(i)
    s_init = c_simp.s_init_cont(i)
    s_incr = c_simp.s_incr_cont(i)
    s_decr = c_simp.s_decr_cont(i)
    
    c_simp.simp.penalty.p = p

    c_simp.simp.optimizer.s_init = s_init
    c_simp.simp.optimizer.s_incr = s_incr
    c_simp.simp.optimizer.s_decr = s_decr

    c_simp.simp.optimizer.model.max_iters = max_iters
    c_simp.simp.optimizer.model.ftol = f_tol
    c_simp.simp.optimizer.model.xtol = x_tol
    c_simp.simp.optimizer.model.grtol = gr_tol

    return
end
