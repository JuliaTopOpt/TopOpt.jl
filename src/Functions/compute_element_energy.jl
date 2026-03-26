"""
    compute_element_energy(cell_energy, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)

Shared kernel for computing element energy (compliance/thermal compliance).

Computes: J = Σ ρ_e * u_e^T Ke u_e
where ρ_e is the penalized density and u_e^T Ke u_e is the element energy.

Gradient: dJ/dx_e = -dρ_e/dx_e * u_e^T Ke u_e

This function is shared between structural compliance and thermal compliance,
differing only in the physics interpretation (strain energy vs thermal energy).
"""
function compute_element_energy(
    cell_energy::Vector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    obj = zero(T)
    grad .= 0
    @inbounds for i in 1:size(cell_dofs, 2)
        cell_energy[i] = zero(T)
        Ke = rawmatrix(Kes[i])
        for w in 1:size(Ke, 2)
            for v in 1:size(Ke, 1)
                cell_energy[i] += u[cell_dofs[v, i]] * Ke[v, w] * u[cell_dofs[w, i]]
            end
        end

        if black[i]
            obj += cell_energy[i]
        elseif white[i]
            if PENALTY_BEFORE_INTERPOLATION
                obj += xmin * cell_energy[i]
            else
                obj += penalty(xmin) * cell_energy[i]
            end
        else
            ρe, dρe = get_ρ_dρ(x[varind[i]], penalty, xmin)
            grad[varind[i]] = -dρe * cell_energy[i]
            obj += ρe * cell_energy[i]
        end
    end

    return obj
end

export compute_element_energy
