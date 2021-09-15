# TODO @params 
mutable struct Displacement{T} <: AbstractFunction{T}
    u::AbstractVector # displacement vector
    dudx_tmp::AbstractVector # directional derivative
    solver::AbstractDisplacementSolver
    global_dofs::AbstractVector{<:Integer}
    fevals::Int
    maxfevals::Int
    reuse::Bool
end

Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Displacement) = println("TopOpt displacement function")

"""
    Displacement()

Construct the Displacement function struct.
"""
function Displacement(solver::AbstractFEASolver; maxfevals = 10^8)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    N = getncells(dh.grid)
    global_dofs = zeros(Int, k)
    # TODO
    u = zeros(T, N)
    dudx_tmp = zeros(T, N)

    return MacroVonMisesStress(u, dudx_tmp, solver, global_dofs, 0, maxfevals, reuse)
end

"""
# Arguments
`x` = design variables

# Returns
displacement vector `u`
"""
function (dp::Displacement{T})(x) where {T}
    @unpack solver, global_dofs, reuse = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1

    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x
    if !reuse
        solver()
    end
    return copy(solver.u)
end

"""
rrule for autodiff. du/dxe = - K \ (dK/dxe * u)

d(u)/d(x_e) = - K^-1 * d(K)/d(x_e) * u
            = - K^-1 * (Σ_ei d(ρ_ei)/d(x_e) * K_ei) * u
            = - K^-1 * [d(ρ_e)/d(x_e) * K_e * u_e](placed in u vector)
"""
function ChainRulesCore.rrule(dp::Displacement, x)
    @unpack reuse, dudx_tmp, solver, global_dofs = dp
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    @unpack K = solver.global_info
    E0 = problem.E
    dp.fevals += 1

    u = dp(x)
    dudx_tmp .= 0
    return u, Δ -> (nothing, begin
        for e in 1:length(Δ)
            ρe, dρe = get_ρ_dρ(x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            dudx_tmp[global_dofs] = - Δ[e] * bcmatrix(Kes[e]) * u[global_dofs] * dρe
        end
        return K \ dudx
    end)
end

