@params mutable struct SIMPResult{T}
    topology::AbstractVector{T}
    objval::T
    convstate
    nsubproblems::Int
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::SIMPResult) = println("TopOpt SIMP result")

function NewSIMPResult(::Type{T}, optimizer, ncells) where {T}
    return SIMPResult(fill(T(NaN), ncells), T(NaN), Nonconvex.ConvergenceState(), 0)
end

"""
The vanilla SIMP algorithm, see [Bendsoe1989](@cite).
"""
@params mutable struct SIMP{T} <: AbstractSIMP
    optimizer
    penalty
    prev_penalty
    solver
    result::SIMPResult{T}
    tracing::Bool
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::SIMP) = println("TopOpt SIMP algorithm")

function SIMP(optimizer, solver, p::T; tracing=true) where T
    penalty = getpenalty(solver)
    prev_penalty = deepcopy(penalty)
    setpenalty!(penalty, p)
    ncells = getncells(solver.problem)
    result = NewSIMPResult(T, optimizer, ncells)
    return SIMP(optimizer, penalty, prev_penalty, solver, result, tracing)
end

Utilities.getpenalty(s::AbstractSIMP) = s.penalty
function Utilities.setpenalty!(s::AbstractSIMP, p::Number)
    penalty = s.penalty
    s.prev_penalty = deepcopy(penalty)
    setpenalty!(penalty, p)
    setpenalty!(s.solver, p)
    return s
end

function (s::SIMP{T, TO})(x0 = s.solver.vars) where {T, TO <: Optimizer}
    setpenalty!(s.solver, s.penalty.p)
    mma_results = s.optimizer(x0)
    update_result!(s, mma_results)
    return s.result
end

function (s::SIMP{T, TO})(workspace::Nonconvex.Workspace) where {T, TO <: Optimizer}
    mma_results = s.optimizer(workspace)
    update_result!(s, mma_results)
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

function update_result!(s::SIMP{T}, mma_results) where T
    # Postprocessing
    @unpack result, optimizer = s
    @unpack problem = s.solver
    @unpack black, white, varind = problem
    nel = getncells(problem)

    update_topology!(result.topology, black, white, mma_results.minimizer, varind)
    result.objval = mma_results.minimum
    if hasproperty(mma_results, :convstate)
        result.convstate = deepcopy(mma_results.convstate)
    end
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
    return topology
end
