mutable struct Volume{
    T,Ts<:AbstractFEASolver,Tc<:AbstractVector{T},Tg<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::Ts
    cellvolumes::Tc
    grad::Tg
    total_volume::T
    fraction::Bool
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::Volume)
    return println(io, "TopOpt volume (fraction) function")
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
    T = eltype(solver.vars)
    cellvolumes = solver.elementinfo.cellvolumes
    # Gradient for all elements (full density vector)
    grad = copy(cellvolumes)
    total_volume = sum(cellvolumes)
    if fraction
        grad ./= total_volume
    end
    return Volume(solver, cellvolumes, grad, total_volume, fraction)
end

function (v::Volume{T})(x::PseudoDensities) where {T}
    vol = compute_volume(v.cellvolumes, x.x)
    return v.fraction ? vol / v.total_volume : vol
end

function ChainRulesCore.rrule(vol::Volume, x::PseudoDensities)
    return vol(x), Δ -> (nothing, Tangent{typeof(x)}(; x=Δ * vol.grad))
end

"""
    compute_volume(cellvolumes::Vector, x)

Compute volume: V = Σ x_e * V_e where x_e is density and V_e is element volume.

Note: x is the full density vector (after projection if using FixedElementProjector).
Black/white elements are handled by the projector - this function receives the
already-projected densities.
"""
function compute_volume(cellvolumes::Vector, x)
    return dot(x, cellvolumes)
end