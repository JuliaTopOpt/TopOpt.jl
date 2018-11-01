mutable struct SIMPResult{T}
    topology::Vector{T}
    objval::T
    fevals::Int
    x_abschange::T
    x_converged::Bool
    f_abschange::T
    f_converged::Bool
    g_residual::T
    g_converged::Bool
    penalty_trace::Vector{Pair{T, Int}}
    nsubproblems::Int
end
function NewSIMPResult(::Type{T}, ncells) where {T}
    SIMPResult(fill(T(NaN), ncells), T(NaN), 0, T(NaN), false, T(NaN), false, T(NaN), false, Pair{T, Int}[], 0)
end

struct SIMP{T, TO, TP} <: AbstractSIMP
    optimizer::TO
    penalty::TP
    result::SIMPResult{T}
    topologies::Vector{Vector{T}}
    tracing::Bool
end
function SIMP(optimizer, p::T, tracing=true) where T
    penalty = optimizer.obj.solver.penalty
    penalty.p = p
    ncells = getncells(optimizer.obj.problem)
    result = NewSIMPResult(T, ncells)
    topologies = Vector{T}[]

    return SIMP{T, typeof(optimizer), typeof(penalty)}(optimizer, penalty, result, topologies, tracing)
end

update_penalty!(s::AbstractSIMP, p::Number) = (s.penalty.p = p)

function (s::SIMP{T, TO})(x0=s.optimizer.obj.solver.vars) where {T, TO<:MMAOptimizer}
    #reset_timer!(to)
    r = @timeit to "SIMP" begin
        prev_l = length(s.topologies)
        prev_fevals = s.optimizer.obj.fevals
        mma_results = s.optimizer(x0)
        update_result!(s, mma_results, prev_l, prev_fevals)
    end
    #display(to)
    return s.result
end

function (s::SIMP{T, TO})(workspace::MMA.MMAWorkspace) where {T, TO<:MMAOptimizer}
    #reset_timer!(to)
    r = @timeit to "SIMP" begin
        prev_l = length(s.topologies)
        prev_fevals = s.optimizer.obj.fevals
        mma_results = s.optimizer(workspace)
        update_result!(s, mma_results, prev_l, prev_fevals)
    end
    #display(to)
    return s.result
end

function update_result!(s::SIMP{T}, mma_results, prev_l, prev_fevals) where T
    # Postprocessing
    #@debug @show mma_results.minimum
    @unpack result, optimizer, topologies = s
    obj = optimizer.obj
    problem = obj.problem
    @unpack black, white, varind = problem
    x_hist = obj.topopt_trace.x_hist    
    nel = getncells(problem)

    if optimizer.obj.tracing
        l = length(x_hist)
        sizehint!(topologies, l)
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
            push!(topologies, copy(topology))
        end
        result.topology .= topologies[end]
    else
        topology = result.topology
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
    result.objval = mma_results.minimum
    new_fevals = obj.fevals - prev_fevals
    result.fevals += new_fevals
    if s.tracing
        push!(result.penalty_trace, (s.penalty.p=>new_fevals))
    end
    if new_fevals > 0
        result.nsubproblems += 1
    end
    result.x_abschange = mma_results.x_abschange
    result.x_converged = mma_results.x_converged
    result.f_abschange = mma_results.f_abschange
    result.f_converged = mma_results.f_converged
    result.g_residual = mma_results.g_residual
    result.g_converged = mma_results.g_converged

    return result
end    
