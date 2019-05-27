@params struct FunctionEvaluations
    obj::Int
    constr
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

@params mutable struct SIMPResult{T, TF <: FunctionEvaluations}
    topology::AbstractVector{T}
    objval::T
    fevals::TF
    convstate
    penalty_trace::AbstractVector{Pair{T, TF}}
    nsubproblems::Int
end

function NewSIMPResult(::Type{T}, optimizer, ncells) where {T}
    FunctionEvaluations(optimizer)
    fevals = zero(FunctionEvaluations(optimizer))
    SIMPResult(fill(T(NaN), ncells), T(NaN), fevals, MMA.ConvergenceState(), Pair{T, typeof(fevals)}[], 0)
end

@params mutable struct SIMP{T} <: AbstractSIMP
    optimizer
    penalty
    prev_penalty
    result::SIMPResult{T}
    tracing::Bool
end

function SIMP(optimizer, p::T; tracing=true) where T
    penalty = getpenalty(optimizer)
    penalty = @set penalty.p = p
    prev_penalty = @set penalty.p = NaN
    ncells = getncells(getsolver(optimizer.obj).problem)
    result = NewSIMPResult(T, optimizer, ncells)

    return SIMP(optimizer, penalty, prev_penalty, result, tracing)
end

Utilities.getpenalty(s::AbstractSIMP) = s.penalty
function Utilities.setpenalty!(s::AbstractSIMP, p::Number)
    penalty = s.penalty
    s.prev_penalty = penalty
    s.penalty = @set penalty.p = p
    setpenalty!(s.optimizer, p)
end

function (s::SIMP{T, TO})(x0=s.optimizer.obj.f.solver.vars, prev_fevals = getfevals(s.optimizer)) where {T, TO<:MMAOptimizer}
    setpenalty!(s.optimizer, s.penalty.p)
    mma_results = s.optimizer(x0)
    update_result!(s, mma_results, prev_fevals)
    return s.result
end

function (s::SIMP{T, TO})(workspace::MMA.Workspace, prev_fevals = getfevals(s.optimizer)) where {T, TO <: MMAOptimizer}
    mma_results = s.optimizer(workspace)
    update_result!(s, mma_results, prev_fevals)
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
        push!(topologies, copy(topology))
    end
    return topologies
end

function update_result!(s::SIMP{T}, mma_results, prev_fevals) where T
    # Postprocessing
    @unpack result, optimizer = s
    @unpack obj = optimizer
    @unpack problem = getsolver(obj)
    @unpack black, white, varind = problem
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
    result.convstate = deepcopy(mma_results.convstate)
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
