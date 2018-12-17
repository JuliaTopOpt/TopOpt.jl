abstract type AbstractPenalty{T} end

mutable struct PowerPenalty{T} <: AbstractPenalty{T}
    p::T
end
(P::PowerPenalty)(x) = x^P.p

mutable struct RationalPenalty{T} <: AbstractPenalty{T}
    p::T
end
(R::RationalPenalty)(x) = sinh(R.p*x)/sinh(R.p)

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)
