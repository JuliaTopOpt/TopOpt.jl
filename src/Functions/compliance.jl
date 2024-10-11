mutable struct Compliance{
    T,TS<:AbstractDisplacementSolver,TC<:AbstractVector{T},TG<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::TS
    cell_comp::TC
    grad::TG
end
Utilities.getpenalty(c::Compliance) = getpenalty(getsolver(c))
Utilities.setpenalty!(c::Compliance, p) = setpenalty!(getsolver(c), p)
Nonconvex.NonconvexCore.getdim(::Compliance) = 1

function Compliance(solver::AbstractDisplacementSolver)
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
cell_compliance = f_e^T * u_e = u_e^T * (ρ_e * Ke) * u_e.
d(cell compliance)/d(x_e) = f_e^T * d(u_e)/d(x_e) = f_e^T * (- K_e^-1 * d(K_e)/d(x_e) * u_e)
                          = - (K_e^(-1) * f_e)^T * d(K_e)/d(x_e) * u_e
                          = - u_e^T * d(ρ_e)/d(x_e) * K_e * u_e
                          = - d(ρ_e)/d(x_e) * cell_compliance
"""
function compute_compliance(
    cell_comp::Vector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    obj = zero(T)
    grad .= 0
    @inbounds for i in 1:size(cell_dofs, 2)
        cell_comp[i] = zero(T)
        Ke = rawmatrix(Kes[i])
        for w in 1:size(Ke, 2)
            for v in 1:size(Ke, 1)
                cell_comp[i] += u[cell_dofs[v, i]] * Ke[v, w] * u[cell_dofs[w, i]]
            end
        end

        if black[i]
            obj += cell_comp[i]
        elseif white[i]
            if PENALTY_BEFORE_INTERPOLATION
                obj += xmin * cell_comp[i]
            else
                obj += penalty(xmin) * cell_comp[i]
            end
        else
            ρe, dρe = get_ρ_dρ(x[varind[i]], penalty, xmin)
            grad[varind[i]] = -dρe * cell_comp[i]
            obj += ρe * cell_comp[i]
        end
    end

    return obj
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
