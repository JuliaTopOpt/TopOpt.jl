abstract type AbstractPenalty{T} end
abstract type AbstractCPUPenalty{T} <: AbstractPenalty{T} end

struct PowerPenalty{T} <: AbstractCPUPenalty{T}
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

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)
