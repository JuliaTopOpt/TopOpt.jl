import ..TopOpt: PseudoDensities, PENALTY_BEFORE_INTERPOLATION

"""
    AbstractPenalty{T}

Abstract type for SIMP-style penalties applied to density variables. `T` is the
numeric type. Concrete subtypes: `PowerPenaltyFun`, `RationalPenaltyFun`,
`SinhPenaltyFun`, `ProjectedPenaltyFun`.
"""
abstract type AbstractPenalty{T} end
abstract type AbstractCPUPenalty{T} <: AbstractPenalty{T} end
"""
    AbstractProjection

Abstract type for projection functions that push densities toward 0/1.
Subtypes: `HeavisideProjectionFun`, `SigmoidProjectionFun`.
"""
abstract type AbstractProjection end

function (P::AbstractCPUPenalty)(x::PseudoDensities{I,<:Any,F}) where {I,F}
    return PseudoDensities{I,true,F}(map(P, x.x))
end

"""
    PowerPenaltyFun(p)

Classic SIMP power penalty: `ρ^p`. `p > 1` penalises intermediate densities.
The most common choice in topology optimization.
"""
mutable struct PowerPenaltyFun{T} <: AbstractCPUPenalty{T}
    p::T
end
(P::PowerPenaltyFun)(x::Real) = x^(P.p)

"""
    RationalPenaltyFun(p)

Rational SIMP penalty: `x / (1 + p * (1 - x))`. Produces a smoother penalty
than `PowerPenaltyFun` for the same exponent.
"""
mutable struct RationalPenaltyFun{T} <: AbstractCPUPenalty{T}
    p::T
end
(R::RationalPenaltyFun)(x::Real) = x / (1 + R.p * (1 - x))

"""
    SinhPenaltyFun(p)

Hyperbolic-sine penalty: `sinh(p*x) / sinh(p)`. An alternative smooth penalty.
"""
mutable struct SinhPenaltyFun{T} <: AbstractCPUPenalty{T}
    p::T
end
(R::SinhPenaltyFun)(x::Real) = sinh(R.p * x) / sinh(R.p)

"""
    ProjectedPenaltyFun(penalty, proj)

Composite penalty that applies a projection (default `HeavisideProjectionFun(10)`)
before the penalty. Pushes the design toward 0/1 and then penalises.
"""
struct ProjectedPenaltyFun{T,Tpen<:AbstractPenalty{T},Tproj} <: AbstractCPUPenalty{T}
    penalty::Tpen
    proj::Tproj
end
function ProjectedPenaltyFun(penalty::AbstractPenalty{T}) where {T}
    return ProjectedPenaltyFun(penalty, HeavisideProjectionFun(10 * one(T)))
end
@inline (P::ProjectedPenaltyFun)(x::Real) = P.penalty(P.proj(x))
@forward_property ProjectedPenaltyFun penalty

function (P::AbstractProjection)(x::PseudoDensities{I,T,F}) where {I,T,F}
    return PseudoDensities{I,T,F}(P(x.x))
end
(P::AbstractProjection)(x::AbstractArray) = map(P, x)

"""
    HeavisideProjectionFun(β)

Heaviside projection with steepness `β`. Larger `β` produces sharper 0/1
transitions. `y = 1 - exp(-β*x) + x*exp(-β)`.
"""
mutable struct HeavisideProjectionFun{T} <: AbstractProjection
    β::T
end
@inline (P::HeavisideProjectionFun)(x::Real) = 1 - exp(-P.β * x) + x * exp(-P.β)

"""
    SigmoidProjectionFun(β)

Sigmoid projection with steepness `β`. `y = 1 / (1 + exp((β+1)*(-x+0.5)))`.
"""
mutable struct SigmoidProjectionFun{T} <: AbstractProjection
    β::T
end
@inline (P::SigmoidProjectionFun)(x::Real) = 1 / (1 + exp((P.β + 1) * (-x + 0.5)))

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)
copy(p::HeavisideProjectionFun) = HeavisideProjectionFun(p.β)
copy(p::SigmoidProjectionFun) = SigmoidProjectionFun(p.β)
copy(p::ProjectedPenaltyFun) = ProjectedPenaltyFun(copy(p.penalty), copy(p.proj))

function Utilities.setpenalty!(P::AbstractPenalty, p)
    P.p = p
    return P
end
function Utilities.setpenalty!(P::ProjectedPenaltyFun, p)
    P.penalty.p = p
    return P
end

function get_ρ(x_e::T, penalty::AbstractPenalty{T}, xmin::T) where {T<:Real}
    if PENALTY_BEFORE_INTERPOLATION
        return density(penalty(x_e), xmin)
    else
        return penalty(density(x_e, xmin))
    end
end

function get_ρ_dρ(x_e::T, penalty::AbstractPenalty{T}, xmin::T) where {T<:Real}
    d = ForwardDiff.Dual{T}(x_e, one(T))
    if PENALTY_BEFORE_INTERPOLATION
        p = density(penalty(d), xmin)
    else
        p = penalty(density(d, xmin))
    end
    g = p.partials[1]
    return p.value, g
end
