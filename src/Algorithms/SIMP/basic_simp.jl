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
whichdevice(s::SIMPResult) = whichdevice(s.topology)

function NewSIMPResult(::Type{T}, ncells) where {T}
    SIMPResult(fill(T(NaN), ncells), T(NaN), 0, T(NaN), false, T(NaN), false, T(NaN), false, Pair{T, Int}[], 0)
end

mutable struct SIMP{T, TO, TP} <: AbstractSIMP
    optimizer::TO
    penalty::TP
    result::SIMPResult{T}
    topologies::Vector{Vector{T}}
    tracing::Bool
end
whichdevice(s::SIMP) = whichdevice(s.optimizer)

function SIMP(optimizer, p::T; tracing=true) where T
    penalty = getpenalty(optimizer)
    penalty = @set penalty.p = p
    ncells = getncells(optimizer.obj.problem)
    result = NewSIMPResult(T, ncells)
    topologies = Vector{T}[]

    return SIMP{T, typeof(optimizer), typeof(penalty)}(optimizer, penalty, result, topologies, tracing)
end

getpenalty(s::AbstractSIMP) = s.penalty
function setpenalty!(s::AbstractSIMP, p::Number)
    penalty = s.penalty
    s.penalty = @set penalty.p = p
    setpenalty!(s.optimizer, p)
end

function (s::SIMP{T, TO})(x0=s.optimizer.obj.solver.vars) where {T, TO<:MMAOptimizer}
    #reset_timer!(to)
    r = @timeit to "SIMP" begin
        setpenalty!(s.optimizer, s.penalty.p)
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
    @unpack obj = optimizer
    @unpack problem = obj
    @unpack black, white, varind = problem
    @unpack x_hist = obj.topopt_trace    
    nel = getncells(problem)

    if optimizer.obj.tracing
        l = length(x_hist)
        sizehint!(topologies, l)
        for i in (prev_l+1) : l
            topology = zeros(T, nel)
            update_topology!(topology, black, white, x_hist[i], varind)
            push!(topologies, topology)
        end
        result.topology .= topologies[end]
    else
        update_topology!(result.topology, black, white, mma_results.minimizer, varind)
    end
    result.objval = mma_results.minimum
    new_fevals = obj.fevals - prev_fevals
    result.fevals += new_fevals
    if s.tracing
        push!(result.penalty_trace, (getpenalty(s).p => new_fevals))
    end
    if new_fevals > 0
        result.nsubproblems += 1
    end
    @unpack x_abschange, x_converged, f_abschange, f_converged, g_residual, g_converged = mma_results
    @pack! result = x_abschange, x_converged, f_abschange, f_converged, g_residual, g_converged

    return result
end    

function update_topology!(topology, black, white, x, varind)
    for i in 1:length(black)
        if black[i]
            topology[i] = 1
        elseif white[i]
            topology[i] = 0
        else
            topology[i] = x[varind[i]]
        end
    end

    return
end
