import ..TopOpt: PseudoDensities, PENALTY_BEFORE_INTERPOLATION

"""
    AbstractPenalty{T}

Abstract type for SIMP-style penalties applied to density variables. `T` is the
numeric type. Concrete subtypes: `PowerPenalty`, `RationalPenalty`, `SinhPenalty`,
`ProjectedPenalty`.
"""
abstract type AbstractPenalty{T} end
abstract type AbstractCPUPenalty{T} <: AbstractPenalty{T} end
"""
    AbstractProjection

Abstract type for projection functions that push densities toward 0/1.
Subtypes: `HeavisideProjection`, `SigmoidProjection`.
"""
abstract type AbstractProjection end

function (P::AbstractCPUPenalty)(x::PseudoDensities{I,<:Any,F}) where {I,F}
    return PseudoDensities{I,true,F}(map(P, x.x))
end

"""
    PowerPenalty(p)

Classic SIMP power penalty: `ρ^p`. `p > 1` penalises intermediate densities.
The most common choice in topology optimization.
"""
mutable struct PowerPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
(P::PowerPenalty)(x::Real) = x^(P.p)

"""
    RationalPenalty(p)

Rational SIMP penalty: `x / (1 + p * (1 - x))`. Produces a smoother penalty
than `PowerPenalty` for the same exponent.
"""
mutable struct RationalPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
(R::RationalPenalty)(x::Real) = x / (1 + R.p * (1 - x))

"""
    SinhPenalty(p)

Hyperbolic-sine penalty: `sinh(p*x) / sinh(p)`. An alternative smooth penalty.
"""
mutable struct SinhPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
(R::SinhPenalty)(x::Real) = sinh(R.p * x) / sinh(R.p)

"""
    ProjectedPenalty(penalty, proj)

Composite penalty that applies a projection (default `HeavisideProjection(10)`)
before the penalty. Pushes the design toward 0/1 and then penalises.
"""
struct ProjectedPenalty{T,Tpen<:AbstractPenalty{T},Tproj} <: AbstractCPUPenalty{T}
    penalty::Tpen
    proj::Tproj
end
function ProjectedPenalty(penalty::AbstractPenalty{T}) where {T}
    return ProjectedPenalty(penalty, HeavisideProjection(10 * one(T)))
end
@inline (P::ProjectedPenalty)(x::Real) = P.penalty(P.proj(x))
@forward_property ProjectedPenalty penalty

function (P::AbstractProjection)(x::PseudoDensities{I,T,F}) where {I,T,F}
    return PseudoDensities{I,T,F}(P(x.x))
end
(P::AbstractProjection)(x::AbstractArray) = map(P, x)

"""
    HeavisideProjection(β)

Heaviside projection with steepness `β`. Larger `β` produces sharper 0/1
transitions. `y = 1 - exp(-β*x) + x*exp(-β)`.
"""
mutable struct HeavisideProjection{T} <: AbstractProjection
    β::T
end
@inline (P::HeavisideProjection)(x::Real) = 1 - exp(-P.β * x) + x * exp(-P.β)

"""
    SigmoidProjection(β)

Sigmoid projection with steepness `β`. `y = 1 / (1 + exp((β+1)*(-x+0.5)))`.
"""
mutable struct SigmoidProjection{T} <: AbstractProjection
    β::T
end
@inline (P::SigmoidProjection)(x::Real) = 1 / (1 + exp((P.β + 1) * (-x + 0.5)))

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)
copy(p::HeavisideProjection) = HeavisideProjection(p.β)
copy(p::SigmoidProjection) = SigmoidProjection(p.β)
copy(p::ProjectedPenalty) = ProjectedPenalty(copy(p.penalty), copy(p.proj))

function Utilities.setpenalty!(P::AbstractPenalty, p)
    P.p = p
    return P
end
function Utilities.setpenalty!(P::ProjectedPenalty, p)
    P.penalty.p = p
    return P
end

function get_ρ(
    x_e::T, penalty::AbstractPenalty{T}, xmin::T
) where {T<:Real}
    if PENALTY_BEFORE_INTERPOLATION
        return density(penalty(x_e), xmin)
    else
        return penalty(density(x_e, xmin))
    end
end

function get_ρ_dρ(
    x_e::T, penalty::AbstractPenalty{T}, xmin::T
) where {T<:Real}
    d = ForwardDiff.Dual{T}(x_e, one(T))
    if PENALTY_BEFORE_INTERPOLATION
        p = density(penalty(d), xmin)
    else
        p = penalty(density(d, xmin))
    end
    g = p.partials[1]
    return p.value, g
end
