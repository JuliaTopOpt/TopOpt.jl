"""
    ConvergenceCriteria

Abstract type for CG convergence criteria. Subtypes: `DefaultCriteria`,
`EnergyCriteria`.
"""
abstract type ConvergenceCriteria end
"""
    DefaultCriteria

Default CG convergence criterion based on the residual norm.
"""
struct DefaultCriteria <: ConvergenceCriteria end
"""
    EnergyCriteria

Energy-based CG convergence criterion, checking the relative energy norm.
Useful for stiff systems where the residual norm is a poor indicator.
"""
mutable struct EnergyCriteria{T} <: ConvergenceCriteria
    energy::T
end
EnergyCriteria() = EnergyCriteria{Float64}(0.0)

const Iterable{Tmat} = Union{CGIterable{Tmat},PCGIterable{<:Any,Tmat}}
function IterativeSolvers.isconverged(
    it::Iterable{<:AbstractMatrixOperator{<:EnergyCriteria}}
)
    conv = it.A.conv
    T = eltype(it.x)
    xtr = dot(it.x, it.r)
    xAx = dot(it.A.f, it.x) - xtr
    energy_change = xAx - conv.energy
    (isnan(energy_change) || isnan(xAx)) && throw(
        DomainError(
            xAx,
            "EnergyCriteria: NaN detected in energy_change=$energy_change or xAx=$xAx",
        ),
    )
    xAx < 0 &&
        throw(DomainError(xAx, "EnergyCriteria: expected non-negative xAx, got $xAx"))
    converged = abs(energy_change) / xAx ≤ it.tol && abs(xtr) / xAx ≤ it.tol
    conv.energy = xAx
    return converged
end
