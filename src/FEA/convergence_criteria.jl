abstract type ConvergenceCriteria end
struct DefaultCriteria <: ConvergenceCriteria end
mutable struct EnergyCriteria{T} <: ConvergenceCriteria
    energy::T
end
EnergyCriteria() = EnergyCriteria{Float64}(0.0)

const Iterable{Tmat} = Union{CGIterable{Tmat}, PCGIterable{<:Any, Tmat}}
function IterativeSolvers.isconverged(it::Iterable{<:AbstractMatrixOperator{<:EnergyCriteria}})
    conv = it.A.conv
    T = eltype(it.x)
    xtr = dot(it.x, it.r)
    xAx = dot(it.A.f, it.x) - xtr
    energy_change = xAx - conv.energy
    @assert !isnan(energy_change) && !isnan(xAx)
    @assert xAx >= 0
    converged = abs(energy_change) / xAx ≤ it.tol && abs(xtr) / xAx ≤ it.tol
    conv.energy = xAx
    return converged
end
