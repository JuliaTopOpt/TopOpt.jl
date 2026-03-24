mutable struct Compliance{
    T,TS<:AbstractFEASolver,TC<:AbstractVector{T},TG<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::TS
    cell_comp::TC
    grad::TG
end
Utilities.getpenalty(c::Compliance) = getpenalty(getsolver(c))
Utilities.setpenalty!(c::Compliance, p) = setpenalty!(getsolver(c), p)
Nonconvex.NonconvexCore.getdim(::Compliance) = 1
getsolver(c::Compliance) = c.solver

function Compliance(solver::AbstractFEASolver)
    # Compliance is only valid for structural (LinearElasticity) problems
    @assert solver.problem isa StiffnessTopOptProblem "Compliance can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(solver.problem))"
    T = eltype(solver.vars)
    cell_comp = zeros(T, getncells(solver.problem.ch.dh.grid))
    grad = copy(cell_comp)
    return Compliance(solver, cell_comp, grad)
end

function (o::Compliance)(x::AbstractVector)
    @warn "A vector input was passed in to the compliance function. It will be assumed to be the filtered, unpenalised and uninterpolated pseudo-densities. Please use the `PseudoDensities` constructor to wrap the input vector to avoid ambiguity."
    return o(PseudoDensities(x))
end
function (o::Compliance{T})(x::PseudoDensities) where {T}
    @unpack cell_comp, solver, grad = o
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata

    penalty = getpenalty(o)
    solver.vars .= x.x
    solver()
    return compute_compliance(
        cell_comp, grad, cell_dofs, Kes, u, black, white, varind, solver.vars, penalty, xmin
    )
end

function ChainRulesCore.rrule(comp::Compliance, x::PseudoDensities)
    out = comp(x)
    out_grad = copy(comp.grad)
    return out, Δ -> (nothing, Tangent{typeof(x)}(; x=out_grad * Δ))
end

"""
    compute_compliance(cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)

Computes structural compliance: J = F^T U = Σ ρ_e * u_e^T Ke u_e
where ρ_e is the penalized density (material stiffness).

Gradient: dJ/dx_e = -u_e^T Ke u_e * dρ_e/dx_e

Uses the shared compute_element_energy kernel.
"""
function compute_compliance(
    cell_comp::Vector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    return compute_element_energy(cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)
end

function compute_inner(inner, u1, u2, solver)
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata
    penalty = getpenalty(solver)
    return compute_inner(
        inner, u1, u2, cell_dofs, Kes, black, white, varind, solver.vars, penalty, xmin
    )
end
function compute_inner(
    inner::AbstractVector{T}, u1, u2, cell_dofs, Kes, black, white, varind, x, penalty, xmin
) where {T}
    obj = zero(T)
    @inbounds for i in 1:size(cell_dofs, 2)
        inner[i] = zero(T)
        cell_comp = zero(T)
        Ke = rawmatrix(Kes[i])
        if !black[i] && !white[i]
            for w in 1:size(Ke, 2)
                for v in 1:size(Ke, 1)
                    cell_comp += u1[cell_dofs[v, i]] * Ke[v, w] * u2[cell_dofs[w, i]]
                end
            end
            ρe, dρe = get_ρ_dρ(x[varind[i]], penalty, xmin)
            inner[varind[i]] = -dρe * cell_comp
        end
    end

    return inner
end
