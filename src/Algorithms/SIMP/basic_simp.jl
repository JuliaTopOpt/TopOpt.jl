struct FunctionEvaluations{TC}
    obj::Int
    constr::TC
end
function FunctionEvaluations(optimizer)
    obj_fevals = getfevals(optimizer.obj)
    constr_fevals = getfevals.(optimizer.constr)
    return FunctionEvaluations(obj_fevals, constr_fevals)
end

Base.zero(t::FunctionEvaluations) = FunctionEvaluations(zero(t.obj), zero.(t.constr))
for f in (:-, :+)
    @eval begin
        function Base.$(f)(f1::FunctionEvaluations, f2::FunctionEvaluations)
            obj_fevals = $(f)(f1.obj, f2.obj)
            constr_fevals = $(f).(f1.constr, f2.constr)
            return FunctionEvaluations(obj_fevals, constr_fevals)
        end
    end
end
Base.broadcastable(f::FunctionEvaluations) = Ref(f)
function Base.all(f, fevals::FunctionEvaluations)
    f(fevals.obj) && all(f, fevals.constr)
end
function Base.any(f, fevals::FunctionEvaluations)
    f(fevals.obj) || any(f, fevals.constr)
end

mutable struct SIMPResult{T, TF <: FunctionEvaluations}
    topology::Vector{T}
    objval::T
    fevals::TF
    x_abschange::T
    x_converged::Bool
    f_abschange::T
    f_converged::Bool
    g_residual::T
    g_converged::Bool
    penalty_trace::Vector{Pair{T, TF}}
    nsubproblems::Int
end
GPUUtils.whichdevice(s::SIMPResult) = whichdevice(s.topology)

function NewSIMPResult(::Type{T}, optimizer, ncells) where {T}
    FunctionEvaluations(optimizer)
    fevals = zero(FunctionEvaluations(optimizer))
    SIMPResult(fill(T(NaN), ncells), T(NaN), fevals, T(NaN), false, T(NaN), false, T(NaN), false, Pair{T, typeof(fevals)}[], 0)
end

mutable struct SIMP{T, TO, TP} <: AbstractSIMP
    optimizer::TO
    penalty::TP
    result::SIMPResult{T}
    tracing::Bool
end
GPUUtils.whichdevice(s::SIMP) = whichdevice(s.optimizer)

function SIMP(optimizer, p::T; tracing=true) where T
    penalty = getpenalty(optimizer)
    penalty = @set penalty.p = p
    ncells = getncells(optimizer.obj.f.problem)
    result = NewSIMPResult(T, optimizer, ncells)

    return SIMP{T, typeof(optimizer), typeof(penalty)}(optimizer, penalty, result, tracing)
end

Utilities.getpenalty(s::AbstractSIMP) = s.penalty
function Utilities.setpenalty!(s::AbstractSIMP, p::Number)
    penalty = s.penalty
    s.penalty = @set penalty.p = p
    setpenalty!(s.optimizer, p)
end

function (s::SIMP{T, TO})(x0=s.optimizer.obj.f.solver.vars) where {T, TO<:MMAOptimizer}
    #reset_timer!(to)
    r = @timeit to "SIMP" begin
        setpenalty!(s.optimizer, s.penalty.p)
        prev_fevals = getfevals(s.optimizer)
        mma_results = s.optimizer(x0)
        update_result!(s, mma_results, prev_fevals)
    end
    #display(to)
    return s.result
end

function (s::SIMP{T, TO})(workspace::MMA.Workspace) where {T, TO<:MMAOptimizer}
    #reset_timer!(to)
    r = @timeit to "SIMP" begin
        prev_fevals = getfevals(s.optimizer)
        mma_results = s.optimizer(workspace)
        update_result!(s, mma_results, prev_fevals)
    end
    #display(to)
    return s.result
end

function get_topologies(problem, trace::TopOptTrace)
    @unpack black, white, varind = problem
    @unpack x_hist = trace
    nel = length(black)
    topologies = Vector{Float64}[]
    topology = zeros(T, nel)
    for i in 1:length(x_hist)
        update_topology!(topology, black, white, x_hist[i], varind)
        push!(topologies, topology)
    end
    return topologies
end

function update_result!(s::SIMP{T}, mma_results, prev_fevals) where T
    # Postprocessing
    #@debug @show mma_results.minimum
    @unpack result, optimizer = s
    @unpack obj = optimizer
    @unpack problem = obj.f
    @unpack black, white, varind = problem
    @unpack x_hist = obj.f.topopt_trace    
    nel = getncells(problem)

    update_topology!(result.topology, black, white, mma_results.minimizer, varind)
    result.objval = mma_results.minimum
    
    new_fevals = getfevals(optimizer)
    extra_fevals = new_fevals - prev_fevals
    result.fevals += extra_fevals
    if s.tracing
        push!(result.penalty_trace, (getpenalty(s).p => extra_fevals))
    end
    if all(x -> x > 0, extra_fevals)
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
