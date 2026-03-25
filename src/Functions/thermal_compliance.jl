"""
    ThermalCompliance{T, TS<:AbstractFEASolver, TC<:AbstractVector{T}, TG<:AbstractVector{T}}

Thermal compliance objective function for heat transfer topology optimization.

# Mathematical Formulation

For steady-state heat conduction:
    -∇·(k(ρ)∇T) = q    in Ω
    T = T_D          on Γ_D (Dirichlet BC)

where:
    k(ρ) = k_min + ρ^p (k_0 - k_min)   (SIMP interpolation)
    q = heat source (constant, NOT penalized)

Thermal compliance: J = Q^T T = ∫ q T dΩ

Gradient (via adjoint method):
    Adjoint: K λ = -∂J/∂T = -Q  →  λ = -T
    dJ/dx_e = -T_e^T Ke T_e · dρ_e/dx_e

CRITICAL: The heat source Q is NOT penalized because it's an external input,
not a material property. Only the conductivity k(ρ) is penalized.

This is the key difference from structural mechanics where body forces depend
on density (self-weight). In heat transfer, q is independent of ρ.

# Usage

```julia
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k, heat_source; Tleft=0.0, Tright=0.0)
solver = FEASolver(DirectSolver, problem; xmin=0.001)
comp = ThermalCompliance(solver)
val = comp(PseudoDensities(ones(length(solver.vars))))
```
"""
mutable struct ThermalCompliance{
    T, TS<:AbstractFEASolver, TC<:AbstractVector{T}, TG<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::TS
    cell_comp::TC
    grad::TG
end

Utilities.getpenalty(tc::ThermalCompliance) = getpenalty(getsolver(tc))
function Utilities.setpenalty!(tc::ThermalCompliance, p)
    return setpenalty!(getsolver(tc), p)
end
Nonconvex.NonconvexCore.getdim(::ThermalCompliance) = 1

getsolver(tc::ThermalCompliance) = tc.solver

function ThermalCompliance(solver::AbstractFEASolver)
    # ThermalCompliance is only valid for heat transfer problems
    @assert solver.problem isa HeatTransferTopOptProblem "ThermalCompliance can only be used with HeatTransferTopOptProblem. Got $(typeof(solver.problem))"
    T = eltype(solver.vars)
    nel = getncells(solver.problem.ch.dh.grid)
    cell_comp = zeros(T, nel)
    grad = copy(cell_comp)
    return ThermalCompliance(solver, cell_comp, grad)
end

function (tc::ThermalCompliance)(x::AbstractVector)
    @warn "A vector input was passed in to the thermal compliance function. It will be assumed to be the filtered, unpenalised and uninterpolated pseudo-densities. Please use the `PseudoDensities` constructor to wrap the input vector to avoid ambiguity."
    return tc(PseudoDensities(x))
end

function (tc::ThermalCompliance{T})(x::PseudoDensities) where {T}
    @unpack cell_comp, grad = tc
    solver = getsolver(tc)
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata

    penalty = getpenalty(tc)
    solver.vars .= x.x
    solver()
    return compute_thermal_compliance(
        cell_comp, grad, cell_dofs, Kes, u, black, white, varind, solver.vars, penalty, xmin
    )
end

function ChainRulesCore.rrule(tc::ThermalCompliance, x::PseudoDensities)
    out = tc(x)
    out_grad = copy(tc.grad)
    return out, Δ -> (nothing, Tangent{typeof(x)}(; x=out_grad * Δ))
end

"""
    compute_thermal_compliance(cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)

Compute thermal compliance and its gradient.

J = Q^T T = Σ_e ρ_e * T_e^T Ke T_e

where:
- T_e: element temperature vector (solution of K T = Q)
- Ke: element conductivity matrix
- ρ_e: penalized density (SIMP interpolation)

Gradient: dJ/dx_e = -dρ_e/dx_e * T_e^T Ke T_e

Uses the shared `compute_element_energy` kernel which computes:
    J = Σ ρ_e * v_e^T Ke v_e

This is mathematically equivalent to structural compliance because both use the
same energy formulation. The key difference is in the physics interpretation:
- Structural: v = displacement, K = stiffness, F = force
- Thermal: v = temperature, K = conductivity, Q = heat source
"""
function compute_thermal_compliance(
    cell_comp::Vector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    return compute_element_energy(cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)
end

export ThermalCompliance