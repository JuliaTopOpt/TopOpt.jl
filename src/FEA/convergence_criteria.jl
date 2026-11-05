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

"""
    IterativeSolvers.isconverged(it::Iterable{<:AbstractMatrixOperator{<:EnergyCriteria}})

Energy-norm-based CG stopping test. The energy `xᵀAx` is recovered from the
iterable state as `bᵀx − rᵀx` (since `r = b − Ax`) and convergence is declared
when both the relative energy change between iterations and the
"correction term" `|rᵀx|` fall below `it.tol` relative to the current energy.

Note: IterativeSolvers 0.9's `cg!` never calls `isconverged` (its `done`
method hardcodes the residual criterion), so `TopOpt.cg_solve!` drives the
iteration manually when an `EnergyCriteria` is attached to the solver, and
stops when *both* this test and the built-in residual test hold (the energy
test is consulted every iteration and updates `conv.energy` as a side effect).
"""
function IterativeSolvers.isconverged(
    it::Iterable{<:AbstractMatrixOperator{<:EnergyCriteria}}
)
    conv = it.A.conv
    xtr = dot(it.x, it.r)
    xAx = dot(it.A.f, it.x) - xtr
    energy_change = xAx - conv.energy
    conv.energy = xAx
    if !isfinite(xAx) || xAx <= 0
        # Degenerate state (zero load, non-finite or non-positive energy):
        # do not declare convergence on the energy test; the residual-based
        # criterion (or `maxiter`) will stop the iteration instead.
        return false
    end
    return abs(energy_change) ≤ it.tol * xAx && abs(xtr) ≤ it.tol * xAx
end
