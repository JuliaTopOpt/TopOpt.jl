"""
    Compliance(solver::AbstractFEASolver)

Differentiable structural compliance objective `J = Fᵀ U = Σ ρ_e u_eᵀ K_e u_e`.

Construct with `Compliance(solver)`. Call as `comp(PseudoDensities(x))` where `x`
is the filtered, optionally projected design. The closed-form gradient
`dJ/dx_e = -u_eᵀ K_e u_e · dρ_e/dx_e` is propagated via a `ChainRulesCore.rrule`.

Only valid for `StiffnessTopOptProblem` with homogeneous Dirichlet BCs.

See [BendsoeSigmund2003](@cite) §2.1 for compliance minimization and
[BendsoeSigmund1999](@cite) for the SIMP interpolation used in the stiffness
assembly.
"""
mutable struct ComplianceFun{
    T,TS<:AbstractFEASolver,TC<:AbstractVector{T},TG<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::TS
    cell_comp::TC
    grad::TG
end
Utilities.getpenalty(c::ComplianceFun) = getpenalty(getsolver(c))
Utilities.setpenalty!(c::ComplianceFun, p) = setpenalty!(getsolver(c), p)
Nonconvex.NonconvexCore.getdim(::ComplianceFun) = 1
getsolver(c::ComplianceFun) = c.solver

function ComplianceFun(solver::AbstractFEASolver)
    solver.problem isa StiffnessTopOptProblem || throw(
        ArgumentError(
            "ComplianceFun can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(solver.problem))",
        ),
    )
    # The closed-form compliance gradient `dJ/dx_e = -dρ_e/dx_e · u_e^T Ke u_e`
    # is only valid for homogeneous Dirichlet BCs. Fail fast on inhomogeneous
    # prescribed displacements rather than silently returning a wrong answer.
    ch = solver.problem.ch
    if any(!=(0), ch.inhomogeneities)
        throw(
            ArgumentError(
                "ComplianceFun assumes homogeneous Dirichlet BCs (prescribed displacement = 0), " *
                "but this problem has nonzero prescribed displacements. The closed-form " *
                "compliance gradient is wrong in that case; use an adjoint-based objective " *
                "or remove the inhomogeneous Dirichlet BCs.",
            ),
        )
    end
    T = eltype(solver.vars)
    cell_comp = zeros(T, getncells(solver.problem.ch.dh.grid))
    grad = copy(cell_comp)
    return ComplianceFun(solver, cell_comp, grad)
end

function (o::ComplianceFun)(x::AbstractVector)
    @warn "A vector input was passed in to the compliance function. It will be assumed to be the filtered, unpenalised and uninterpolated pseudo-densities. Please use the `PseudoDensities` constructor to wrap the input vector to avoid ambiguity."
    return o(PseudoDensities(x))
end
function (o::ComplianceFun{T})(x::PseudoDensities) where {T}
    @unpack cell_comp, solver, grad = o
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes = elementinfo
    @unpack cell_dofs = metadata

    penalty = getpenalty(o)
    solver.vars .= x.x
    solver()
    return compute_compliance(
        cell_comp, grad, cell_dofs, Kes, u, solver.vars, penalty, xmin
    )
end

function ChainRulesCore.rrule(comp::ComplianceFun, x::PseudoDensities)
    out = comp(x)
    out_grad = copy(comp.grad)
    return out, Δ -> (nothing, Tangent{typeof(x)}(; x=out_grad * ChainRulesCore.unthunk(Δ)))
end

"""
    compute_compliance(cell_comp, grad, cell_dofs, Kes, u, x, penalty, xmin)

Computes structural compliance: J = F^T U = Σ ρ_e * u_e^T Ke u_e
where ρ_e is the penalized density (material stiffness).

Gradient: dJ/dx_e = -u_e^T Ke u_e * dρ_e/dx_e

Note: x is the full density vector (after projection if using FixedElementProjector).
Uses the shared compute_element_energy kernel.
"""
function compute_compliance(
    cell_comp::Vector{T}, grad, cell_dofs, Kes, u, x, penalty, xmin
) where {T}
    return compute_element_energy(cell_comp, grad, cell_dofs, Kes, u, x, penalty, xmin)
end

function compute_inner(inner, u1, u2, solver)
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes = elementinfo
    @unpack cell_dofs = metadata
    penalty = getpenalty(solver)
    return compute_inner(inner, u1, u2, cell_dofs, Kes, solver.vars, penalty, xmin)
end
function compute_inner(
    inner::AbstractVector{T}, u1, u2, cell_dofs, Kes, x, penalty, xmin
) where {T}
    obj = zero(T)
    @inbounds for i in axes(cell_dofs, 2)
        inner[i] = zero(T)
        cell_comp = zero(T)
        Ke = rawmatrix(Kes[i])
        for w in axes(Ke, 2)
            for v in axes(Ke, 1)
                cell_comp += u1[cell_dofs[v, i]] * Ke[v, w] * u2[cell_dofs[w, i]]
            end
        end
        ρe, dρe = get_ρ_dρ(x[i], penalty, xmin)
        inner[i] = -dρe * cell_comp
    end

    return inner
end
