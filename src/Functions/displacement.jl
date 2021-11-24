@params mutable struct Displacement{T} <: AbstractFunction{T}
    u::AbstractVector{T} # displacement vector
    dudx_tmp::AbstractVector # directional derivative
    solver::AbstractDisplacementSolver
    global_dofs::AbstractVector{<:Integer}
    fevals::Int
    maxfevals::Int
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
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    dudx_tmp = zeros(T, length(solver.vars))
    return Displacement(u, dudx_tmp, solver, global_dofs, 0, maxfevals)
end

"""
# Arguments
`x` = design variables

# Returns
displacement vector `u`
"""
function (dp::Displacement{T})(x::AbstractVector) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x
    solver()
    return copy(solver.u)
end

function (dp::Displacement{T})(K::SparseMatrixCSC) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x
    solver()
    return copy(solver.u)
end

"""
rrule for autodiff.
    
du/dxe = -K^-1 * dK/dxe * u
d(u)/d(x_e) = - K^-1 * d(K)/d(x_e) * u
            = - K^-1 * (Σ_ei d(ρ_ei)/d(x_e) * K_ei) * u
            = - K^-1 * [d(ρ_e)/d(x_e) * K_e * u]
d(u)/d(x_e)' * Δ = -d(ρ_e)/d(x_e) * u' * K_e * (K^-1 * Δ)

where d(u)/d(x) ∈ (nDof x nCell); d(u)/d(x)^T * Δ = (nCell x nDof) * (nDof x 1) -> nCell x 1
"""
function ChainRulesCore.rrule(dp::Displacement, x::AbstractVector)
    @unpack dudx_tmp, solver, global_dofs = dp
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    # Forward-pass
    # Cholesky factorisation
    u = dp(x)
    return u, Δ -> begin # v
        solver.rhs .= Δ
        solver(reuse_chol = true, assemble_f = false)
        dudx_tmp .= 0
        for e in 1:length(x)
            _, dρe = get_ρ_dρ(x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            Keu = bcmatrix(Kes[e]) * u[global_dofs]
            dudx_tmp[e] = -dρe * dot(Keu, solver.lhs[global_dofs])
        end
        return nothing, dudx_tmp # J1' * v, J2' * v
    end
end

function ChainRulesCore.rrule(dp::Displacement, K::SparseMatrixCSC)
    val, pb = Zygote.gradient(\, K, dp.f)
    return val, Δ -> begin
        temp  = pb(Δ)
        return (NoTangent(), temp[2])
    end
end
