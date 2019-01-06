abstract type AbstractPenalty{T} end
abstract type AbstractCPUPenalty{T} <: AbstractPenalty{T} end
abstract type AbstractGPUPenalty{T} <: AbstractPenalty{T} end

GPUUtils.whichdevice(::AbstractCPUPenalty) = CPU()
GPUUtils.whichdevice(::AbstractGPUPenalty) = GPU()

CUDAnative.pow(d::TD, p::AbstractFloat) where {T, TV, TD <: ForwardDiff.Dual{T, TV, 1}} = ForwardDiff.Dual{T}(CUDAnative.pow(d.value, p), p * d.partials[1] * CUDAnative.pow(d.value, p - 1))

struct PowerPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
@inline (P::PowerPenalty)(x) = x^(P.p)

struct GPUPowerPenalty{T} <: AbstractGPUPenalty{T}
    p::T
end
@inline (P::GPUPowerPenalty)(x) = CUDAnative.pow(x, P.p)

struct RationalPenalty{T} <: AbstractCPUPenalty{T}
    p::T
end
@inline (R::RationalPenalty)(x) = sinh(R.p*x)/sinh(R.p)

struct GPURationalPenalty{T} <: AbstractGPUPenalty{T}
    p::T
end
@inline (P::GPURationalPenalty)(x) = CUDAnative.sinh(R.p*x)/CUDAnative.sinh(R.p)

import Base: copy
copy(p::TP) where {TP<:AbstractPenalty} = TP(p.p)

for T in (:PowerPenalty, :RationalPenalty)
    fname = Symbol(:GPU, T)
    @eval @inline CuArrays.cu(p::$T) = $fname(p.p)
end
CuArrays.cu(p::AbstractGPUPenalty) = p
