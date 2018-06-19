abstract type TopOptAlgorithm end

mutable struct SIMPResult{T}
    topology::Vector{T}
    objval::T
    x_abschange::T
    x_converged::Bool
    f_abschange::T
    f_converged::Bool
    g_residual::T
    g_converged::Bool
end
struct SIMP{T, TO, TP} <: TopOptAlgorithm 
    optimizer::TO
    penalty::TP
    result::SIMPResult{T}
    topologies::Vector{Vector{T}}
end
function SIMP(optimizer, p::T) where T
    penalty = optimizer.obj.solver.penalty
    penalty.p = p
    topology = fill(T(NaN), getncells(optimizer.obj.problem.ch.dh.grid))
    objval = T(NaN)
    result = SIMPResult(topology, objval, T(NaN), false, T(NaN), false, T(NaN), false)
    topologies = Vector{T}[]

    return SIMP{T, typeof(optimizer), typeof(penalty)}(optimizer, penalty, result, topologies)
end

update_penalty!(s::SIMP, p::Number) = (s.penalty.p = p)

function (s::SIMP{T, TO})(x0=s.optimizer.obj.solver.vars) where {T, TO<:MMAOptimizer}
    prev_l = length(s.topologies)
    mma_results = s.optimizer(x0)
    update_result!(s, mma_results, prev_l)
    return s.result
end

function update_result!(s::SIMP{T}, mma_results, prev_l) where T
    # Postprocessing
    nel = getncells(s.optimizer.obj.problem.ch.dh.grid)
    varind = s.optimizer.obj.problem.varind

    x_hist = s.optimizer.obj.topopt_trace.x_hist
    black = s.optimizer.obj.problem.black
    white = s.optimizer.obj.problem.white
    
    if s.optimizer.obj.tracing
        l = length(x_hist)
        sizehint!(s.topologies, l)
        for i in (prev_l+1) : l
            topology = zeros(T, nel)
            for j in 1:nel
                if black[j]
                    topology[j] = 1
                elseif white[j]
                    topology[j] = 0
                else
                    topology[j] = x_hist[i][varind[j]]
                end
            end
            push!(s.topologies, copy(topology))
        end
        s.result.topology .= s.topologies[end]
    else
        topology = s.result.topology
        minimizer = mma_results.minimizer
        for i in 1:nel
            if black[i]
                topology[i] = 1
            elseif white[i]
                topology[i] = 0
            else
                topology[i] = minimizer[varind[i]]
            end
        end
    end
    s.result.objval = mma_results.minimum

    s.result.x_abschange = mma_results.x_abschange
    s.result.x_converged = mma_results.x_converged
    s.result.f_abschange = mma_results.f_abschange
    s.result.f_converged = mma_results.f_converged
    s.result.g_residual = mma_results.g_residual
    s.result.g_converged = mma_results.g_converged

    return
end    
