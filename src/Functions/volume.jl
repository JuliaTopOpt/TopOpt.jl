@params mutable struct Volume{T, dim} <: AbstractFunction{T}
    problem::StiffnessTopOptProblem{dim, T}
    solver::AbstractFEASolver
    cellvolumes::AbstractVector{T}
    grad::AbstractVector{T}
    total_volume::T
    fixed_volume::T
    tracing::Bool
	topopt_trace::TopOptTrace{T}
    fraction::Bool
    fevals::Int
    maxfevals::Int
    cheqfilter
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Volume) = println("TopOpt volume (fraction) function")
Nonconvex.getdim(::Volume) = 1
@inline function Base.getproperty(vf::Volume, f::Symbol)
    f === :reuse && return false
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::Volume, f::Symbol, v)
    f === :reuse && return false
    return setfield!(vf, f, v)
end

function project(c::IneqConstraint{<:Any, <:Volume}, x)
    V, f = c.s, c.f
    cellvolumes = f.cellvolumes
    if f.fraction
        V = V * f.total_volume        
    end
    inds = sortperm(x, rev=true)
    total = zero(V)
    i = 0
    while i <= length(inds)
        i += 1
        ind = inds[i]
        _total = total + cellvolumes[ind]
        _total >= V && break
        total = _total
    end
    x = zeros(eltype(x), length(x))
    x[inds[1:i]] .= 1
    return x
end

function Volume(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractFEASolver, ::Type{TI} = Int; filterT = nothing, preproj = nothing, postproj = nothing, rmin = T(0), fraction = true, tracing = true, maxfevals = 10^8) where {dim, T, TI}
    rmin == 0 && filterT !== nothing && throw("Cannot use a filter radius of 0 in a density filter.")
    dh = problem.ch.dh
    varind = problem.varind
    black = problem.black
    white = problem.white
    vars = solver.vars
    cellvolumes = solver.elementinfo.cellvolumes

    if filterT isa Nothing
        cheqfilter = SensFilter(Val(false), solver, rmin)
    elseif filterT isa SensFilter
        cheqfilter = SensFilter(Val(false), solver, rmin)
    else
        if preproj isa Nothing && postproj isa Nothing
            cheqfilter = DensityFilter(Val(true), solver, rmin)
        else
            cheqfilter = ProjectedDensityFilter(DensityFilter(Val(true), solver, rmin), preproj, postproj)
        end
    end

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
    return Volume(problem, solver, cellvolumes, grad, total_volume, fixed_volume, tracing, TopOptTrace{T, TI}(), fraction, 0, maxfevals, cheqfilter)
end
function (v::Volume{T})(x, grad = nothing) where {T}
    varind = v.problem.varind
    black = v.problem.black
    white = v.problem.white
    cellvolumes = v.cellvolumes
    total_volume = v.total_volume
    fixed_volume = v.fixed_volume
    fraction = v.fraction
    v.fevals += 1

    tracing = v.tracing
    topopt_trace = v.topopt_trace
    dh = v.problem.ch.dh
    xmin = v.solver.xmin

    if v.cheqfilter isa AbstractDensityFilter
        fx = v.cheqfilter(x)
    else
        fx = x
    end
    vol = compute_volume(cellvolumes, fx, fixed_volume, varind, black, white)

    constrval = fraction ? vol / total_volume : vol
    if grad !== nothing
        grad .= v.grad
        if v.cheqfilter isa AbstractDensityFilter
            grad .= TopOpt.jtvp!(similar(grad), v.cheqfilter, x, grad)
        end
    end
    if tracing
        push!(topopt_trace.v_hist, vol/total_volume)
    end

    return constrval
end
function ChainRulesCore.rrule(vol::Volume, x)
    grad = similar(vol.grad)
    return vol(x, grad), Δ -> (nothing, Δ * grad)
end

function compute_volume(cellvolumes::Vector, x, fixed_volume, varind, black, white)
    vol = fixed_volume
    for i in 1:length(cellvolumes)
        if !(black[i]) && !(white[i])
            #vol += density(x[varind[i]], xmin)*cellvolumes[i]
            vol += x[varind[i]]*cellvolumes[i]
        end
    end
    return vol
end
