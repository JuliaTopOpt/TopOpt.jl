abstract type AbstractPenalty{T} end
abstract type AbstractCPUPenalty{T} <: AbstractPenalty{T} end
abstract type AbstractProjection end

mutable struct PowerPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
@inline (P::PowerPenalty)(x) = x^(P.p)

mutable struct RationalPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
@inline (R::RationalPenalty)(x) = x / (1 + R.p * (1 - x))

mutable struct SinhPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
@inline (R::SinhPenalty)(x) = sinh(R.p*x)/sinh(R.p)

struct ProjectedPenalty{T, Tpen <: AbstractPenalty{T}, Tproj} <: AbstractCPUPenalty{T}
    penalty::Tpen
    proj::Tproj
end
function ProjectedPenalty(penalty::AbstractPenalty{T}) where {T}
    return ProjectedPenalty(penalty, HeavisideProjection(10*one(T)))
end
@inline (P::ProjectedPenalty)(x) = P.penalty(P.proj(x))
@forward_property ProjectedPenalty penalty

mutable struct HeavisideProjection{T} <: AbstractProjection
    β::T
end
@inline (P::HeavisideProjection)(x) = 1 - exp(-P.β*x) + x * exp(-P.β)
mutable struct SigmoidProjection{T} <: AbstractProjection
    β::T
end
@inline (P::SigmoidProjection)(x) = 1 / (1 + exp((P.β + 1) * (-x + 0.5)))

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
