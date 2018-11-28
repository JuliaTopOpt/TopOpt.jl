abstract type AbstractPenalty{T} end

CUDAnative.pow(d::TD, p::AbstractFloat) where {T, TV, TD <: ForwardDiff.Dual{T, TV, 1}} = ForwardDiff.Dual{T}(CUDAnative.pow(d.value, p), p * d.partials[1] * CUDAnative.pow(d.value, p - 1))

struct PowerPenalty{T} <: AbstractPenalty{T}
    p::T
end
@inline (P::PowerPenalty)(x) = x^(P.p)
struct GPUPowerPenalty{T} <: AbstractPenalty{T}
    p::T
end
@inline (P::GPUPowerPenalty)(x) = CUDAnative.pow(x, P.p)

struct RationalPenalty{T} <: AbstractPenalty{T}
    p::T
end
@inline (R::RationalPenalty)(x) = sinh(R.p*x)/sinh(R.p)

struct GPURationalPenalty{T} <: AbstractPenalty{T}
    p::T
end
@inline (P::GPURationalPenalty)(x) = CUDAnative.pow(x, P.p)

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)
