@params mutable struct ProjectedVolume{T, dim} <: AbstractFunction{T}
    problem::StiffnessTopOptProblem{dim, T}
    solver::AbstractFEASolver
    filter
    proj
    cellvolumes::AbstractVector{T}
    grad::AbstractVector{T}
    total_volume::T
    fixed_volume::T
    tracing::Bool
	topopt_trace::TopOptTrace{T}
    fraction::Bool
    fevals::Int
    maxfevals::Int
end
TopOpt.dim(::ProjectedVolume) = 1
@inline function Base.getproperty(vf::ProjectedVolume, f::Symbol)
    f === :reuse && return false
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::ProjectedVolume, f::Symbol, v)
    f === :reuse && return false
    return setfield!(vf, f, v)
end

function ProjectedVolume(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractFEASolver, filter, proj, ::Type{TI} = Int; fraction = true, tracing = true, maxfevals = 10^8) where {dim, T, TI}
    dh = problem.ch.dh
    varind = problem.varind
    black = problem.black
    white = problem.white
    vars = solver.vars
    cellvolumes = solver.elementinfo.cellvolumes

    grad = zeros(T, length(vars))
    for (i, cell) in enumerate(CellIterator(dh))
        if !(black[i]) && !(white[i])
            grad[varind[i]] = cellvolumes[i]
        end
    end
    total_volume = sum(cellvolumes)
    fixed_volume = dot(black, cellvolumes)
    if fraction
        grad ./= total_volume
    end
    return ProjectedVolume(problem, solver, filter, proj, cellvolumes, grad, total_volume, fixed_volume, tracing, TopOptTrace{T, TI}(), fraction, 0, maxfevals)
end
function (v::ProjectedVolume{T})(x, grad = nothing) where {T}
    @unpack varind, black, white = v.problem
    @unpack cellvolumes, total_volume, fixed_volume, fraction, filter, proj, tracing, topopt_trace = v
    v.fevals += 1
    dh = v.problem.ch.dh
    xmin = v.solver.xmin

    if filter isa AbstractDensityFilter
        filtered_x = filter(copy(x))
    end
    vol = zero(T)
    if grad !== nothing
        for i in 1:length(black)
            if black[i]
                vol += cellvolumes[i]
            elseif white[i]
                nothing
            else
                d = ForwardDiff.Dual{T}(filtered_x[varind[i]], one(T))
                p = proj(d)
                grad[varind[i]] = -p.partials[1]
                vol += p.value * cellvolumes[i]
            end
        end
        if filter isa AbstractDensityFilter
            grad .= TopOpt.jtvp!(similar(grad), filter, x, grad)
        end
        v.grad .= grad
    end
    if fraction
        constrval = vol / total_volume
        if grad !== nothing
            grad ./= total_volume
            v.grad .= grad
        end
    else
        constrval = vol
    end
    if tracing
        push!(topopt_trace.v_hist, vol/total_volume)
    end

    return constrval
end
