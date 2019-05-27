using ..CUDASupport
using ..TopOpt: @init_cuda, CPU, GPU
@init_cuda()
import ..TopOpt: whichdevice
using ..GPUUtils, ForwardDiff

abstract type AbstractGPUPenalty{T} <: AbstractPenalty{T} end

whichdevice(::AbstractCPUPenalty) = CPU()
whichdevice(::AbstractGPUPenalty) = GPU()

CUDAnative.pow(d::TD, p::AbstractFloat) where {T, TV, TD <: ForwardDiff.Dual{T, TV, 1}} = ForwardDiff.Dual{T}(CUDAnative.pow(d.value, p), p * d.partials[1] * CUDAnative.pow(d.value, p - 1))

struct GPUPowerPenalty{T} <: AbstractGPUPenalty{T}
    p::T
end
@inline (P::GPUPowerPenalty)(x) = CUDAnative.pow(x, P.p)

struct GPURationalPenalty{T} <: AbstractGPUPenalty{T}
    p::T
end
@inline (P::GPURationalPenalty)(x) = x / (1 + R.p * (1 - x))

struct GPUSinhPenalty{T} <: AbstractGPUPenalty{T}
    p::T
end
@inline (P::GPUSinhPenalty)(x) = CUDAnative.sinh(R.p*x)/CUDAnative.sinh(R.p)

for T in (:PowerPenalty, :RationalPenalty)
    fname = Symbol(:GPU, T)
    @eval @inline CuArrays.cu(p::$T) = $fname(p.p)
end
CuArrays.cu(p::AbstractGPUPenalty) = p

@define_cu(IterativeSolvers.CGStateVariables, :u, :r, :c)
whichdevice(ra::RaggedArray) = whichdevice(ra.offsets)
@define_cu(RaggedArray, :offsets, :values)
