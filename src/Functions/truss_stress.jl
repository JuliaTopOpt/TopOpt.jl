mutable struct TrussStress{
    T,Ts<:AbstractVector{T},Tu<:Displacement,Tt<:AbstractVector{<:AbstractMatrix{T}}
} <: AbstractFunction{T}
    σ::Ts # stress vector, axial stress per cell
    u_fn::Tu
    transf_matrices::Tt
    fevals::Int
    maxfevals::Int
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::TrussStress)
    return println("TopOpt truss stress function")
end

"""
    TrussStress(solver; maxfevals=10^8)

Construct the TrussStress function struct.
"""
function TrussStress(solver::AbstractFEASolver; maxfevals=10^8)
    T = eltype(solver.u)
    dim = TopOptProblems.getdim(solver.problem)
    dh = solver.problem.ch.dh
    N = getncells(dh.grid)
    σ = zeros(T, N)
    transf_matrices = Matrix{T}[]
    u_fn = Displacement(solver; maxfevals)
    R = zeros(T, (2, 2 * dim))
    for (cellidx, cell) in enumerate(CellIterator(dh))
        u, v = cell.coords[1], cell.coords[2]
        # R ∈ 2 x (2*dim)
        R_coord = compute_local_axes(u, v)
        fill!(R, 0.0)
        R[1, 1:dim] = R_coord[:, 1]
        R[2, (dim + 1):(2 * dim)] = R_coord[:, 2]
        push!(transf_matrices, copy(R))
    end
    return TrussStress(σ, u_fn, transf_matrices, 0, maxfevals)
end

"""
# Arguments
`x` = design variables

# Returns
displacement vector `σ`, compressive stress < 0, tensile stress > 0
"""
function (ts::TrussStress{T})(x::PseudoDensities) where {T}
    @unpack σ, transf_matrices, u_fn = ts
    @unpack global_dofs, solver = u_fn
    @unpack penalty, problem, xmin = solver
    dh = getdh(problem)
    ts.fevals += 1
    u = u_fn(x)
    As = getA(problem)
    @unpack Kes = solver.elementinfo
    for e in 1:length(x)
        # Ke = R' * K_local * R
        # F = R * (R' * K_local * R) * u
        celldofs!(global_dofs, dh, e)
        σ[e] = -(transf_matrices[e] * Kes[e] * u.u[global_dofs])[1] / As[e]
    end
    return copy(σ)
end

# TODO complete
# """
# rrule for autodiff.

# du/dxe = -K^-1 * dK/dxe * u
# d(u)/d(x_e) = - K^-1 * d(K)/d(x_e) * u
#             = - K^-1 * (Σ_ei d(ρ_ei)/d(x_e) * K_ei) * u
#             = - K^-1 * [d(ρ_e)/d(x_e) * K_e * u]
# d(u)/d(x_e)' * Δ = -d(ρ_e)/d(x_e) * u' * K_e * (K^-1 * Δ)
# """
# function ChainRulesCore.rrule(dp::TrussStress, x)
#     @unpack dudx_tmp, solver, global_dofs = dp
#     @unpack penalty, problem, u, xmin = solver
#     dh = getdh(problem)
#     @unpack Kes = solver.elementinfo
#     # Forward-pass
#     # Cholesky factorisation
#     u = dp(x)
#     return u, Δ -> begin # v
#         solver.rhs .= Δ
#         solver(reuse_fact = true, assemble_f = false)
#         dudx_tmp .= 0
#         for e in 1:length(x)
#             _, dρe = get_ρ_dρ(x[e], penalty, xmin)
#             celldofs!(global_dofs, dh, e)
#             Keu = bcmatrix(Kes[e]) * u[global_dofs]
#             dudx_tmp[e] = -dρe * dot(Keu, solver.lhs[global_dofs])
#         end
#         return nothing, dudx_tmp # J1' * v, J2' * v
#     end
# end
