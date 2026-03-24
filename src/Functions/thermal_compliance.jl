mutable struct ThermalCompliance{
    T,TS<:AbstractFEASolver,TC<:AbstractVector{T},TG<:AbstractVector{T}
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

Computes thermal compliance: J = T^T K T = Σ ρ_e * u_e^T Ke u_e
where ρ_e is the penalized density (material thermal conductivity).

Gradient: dJ/dρ_e = -u_e^T Ke u_e * dρ_e/dx_e
For minimization with SIMP, this drives material toward low-conductivity regions.
"""
function compute_thermal_compliance(
    cell_comp::Vector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    obj = zero(T)
    grad .= 0
    @inbounds for i in 1:size(cell_dofs, 2)
        cell_comp[i] = zero(T)
        Ke = rawmatrix(Kes[i])
        # Compute element thermal compliance: u_e^T Ke u_e
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

"""
    MeanTemperature{T,TS<:AbstractFEASolver} <: AbstractFunction{T}

Computes the mean temperature over the domain. Useful for heat sink design
where we want to minimize the average temperature.
"""
mutable struct MeanTemperature{T,TS<:AbstractFEASolver,Tv<:AbstractVector{T}} <: AbstractFunction{T}
    solver::TS
    cell_volumes::Tv
    total_volume::T
    fixed_volume::T
    grad::Tv
end

Nonconvex.NonconvexCore.getdim(::MeanTemperature) = 1

function MeanTemperature(solver::AbstractFEASolver)
    # MeanTemperature is only valid for heat transfer problems
    @assert solver.problem isa HeatTransferTopOptProblem "MeanTemperature can only be used with HeatTransferTopOptProblem. Got $(typeof(solver.problem))"
    T = eltype(solver.vars)
    problem = solver.problem
    cell_volumes = solver.elementinfo.cellvolumes
    total_volume = sum(cell_volumes)
    fixed_volume = sum(cell_volumes[problem.black])
    grad = zeros(T, length(solver.vars))
    return MeanTemperature(solver, cell_volumes, total_volume, fixed_volume, grad)
end

function (mt::MeanTemperature)(x::PseudoDensities)
    solver = mt.solver
    solver.vars .= x.x
    solver()
    temp = solver.u
    # Mean temperature weighted by element volume
    cell_temps = compute_cell_temperatures(temp, solver.elementinfo)
    return compute_mean_temperature(
        cell_temps, x.x, mt.cell_volumes, mt.fixed_volume,
        solver.problem.varind, solver.problem.black, solver.problem.white
    )
end

function ChainRulesCore.rrule(mt::MeanTemperature, x::PseudoDensities)
    val = mt(x)
    # Gradient computation would require solving adjoint problem
    # For now, use finite differences or Zygote
    return val, Δ -> (nothing, Tangent{typeof(x)}(; x=mt.grad * Δ))
end

"""
    compute_cell_temperatures(u, elementinfo)

Computes the average temperature in each element from the nodal temperatures.
"""
function compute_cell_temperatures(u::AbstractVector, elementinfo::ElementFEAInfo)
    T = eltype(u)
    cell_dofs = elementinfo.metadata.cell_dofs
    ncells = size(cell_dofs, 2)
    cell_temps = zeros(T, ncells)
    for i in 1:ncells
        ndofs_cell = size(cell_dofs, 1)
        temp_sum = zero(T)
        for j in 1:ndofs_cell
            temp_sum += u[cell_dofs[j, i]]
        end
        cell_temps[i] = temp_sum / ndofs_cell
    end
    return cell_temps
end

function compute_mean_temperature(cell_temps, x, cell_volumes, fixed_volume, varind, black, white)
    T = eltype(cell_temps)
    total_temp_volume = zero(T)
    total_volume = sum(cell_volumes)

    for i in 1:length(cell_volumes)
        if black[i]
            total_temp_volume += cell_temps[i] * cell_volumes[i]
        elseif white[i]
            total_temp_volume += cell_temps[i] * cell_volumes[i] * 0.001  # xmin approximation
        else
            total_temp_volume += cell_temps[i] * cell_volumes[i] * x[varind[i]]
        end
    end

    return total_temp_volume / total_volume
end

export ThermalCompliance, MeanTemperature, compute_cell_temperatures