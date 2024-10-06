mutable struct Volume{
    T,Ts<:AbstractFEASolver,Tc<:AbstractVector{T},Tg<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::Ts
    cellvolumes::Tc
    grad::Tg
    total_volume::T
    fixed_volume::T
    fraction::Bool
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Volume)
    return println("TopOpt volume (fraction) function")
end
Nonconvex.NonconvexCore.getdim(::Volume) = 1

function project(f::Volume, V, x)
    cellvolumes = f.cellvolumes
    if f.fraction
        V = V * f.total_volume
    end
    inds = sortperm(x; rev=true)
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

function Volume(solver::AbstractFEASolver; fraction=true)
    problem = solver.problem
    dh = problem.ch.dh
    varind = problem.varind
    black = problem.black
    white = problem.white
    vars = solver.vars
    cellvolumes = solver.elementinfo.cellvolumes
    T = eltype(solver.vars)
    grad = zeros(T, length(vars))
    for (i, _) in enumerate(CellIterator(dh))
        if !(black[i]) && !(white[i])
            grad[varind[i]] = cellvolumes[i]
        end
    end
    total_volume = sum(cellvolumes)
    fixed_volume = dot(black, cellvolumes)
    if fraction
        grad ./= total_volume
    end
    return Volume(solver, cellvolumes, grad, total_volume, fixed_volume, fraction)
end
function (v::Volume{T})(x::PseudoDensities) where {T}
    problem = v.solver.problem
    varind = problem.varind
    black = problem.black
    white = problem.white
    cellvolumes = v.cellvolumes
    total_volume = v.total_volume
    fixed_volume = v.fixed_volume
    fraction = v.fraction
    vol = compute_volume(cellvolumes, x, fixed_volume, varind, black, white)
    return fraction ? vol / total_volume : vol
end
function ChainRulesCore.rrule(vol::Volume, x::PseudoDensities)
    return vol(x), Δ -> (nothing, Tangent{typeof(x)}(; x=Δ * vol.grad))
end

function compute_volume(cellvolumes::Vector, x, fixed_volume, varind, black, white)
    vol = fixed_volume
    for i in 1:length(cellvolumes)
        if !(black[i]) && !(white[i])
            vol += x[varind[i]] * cellvolumes[i]
        end
    end
    return vol
end
