@params mutable struct Displacement{T} <: AbstractFunction{T}
    u::AbstractVector{T} # displacement vector
    dudx_tmp::AbstractVector # directional derivative
    solver::AbstractDisplacementSolver
    global_dofs::AbstractVector{<:Integer}
    fevals::Int
    maxfevals::Int
end

@params mutable struct HyperelasticDisplacement{T} <: AbstractFunction{T}
    u::AbstractVector{T} # displacement vector
    F::AbstractVector # deformation gradient tensor
    dudx_tmp::AbstractVector # directional derivative
    solver::AbstractHyperelasticSolver
    global_dofs::AbstractVector{<:Integer}
    fevals::Int
    maxfevals::Int
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::Displacement)
    return println("TopOpt displacement function")
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::HyperelasticDisplacement)
    return println("TopOpt displacement function for hyperelastic strain regimes")
end

struct DisplacementResult{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    u::A
end

Base.length(u::DisplacementResult) = length(u.u)
Base.size(u::DisplacementResult, i...) = size(u.u, i...)
Base.getindex(u::DisplacementResult, i...) = u.u[i...]
Base.sum(u::DisplacementResult) = sum(u.u)
LinearAlgebra.dot(u::DisplacementResult, weights::AbstractArray) = dot(u.u, weights)

"""
    Displacement()

Construct the Displacement function struct.
"""
function Displacement(solver::AbstractFEASolver; maxfevals=10^8)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    dudx_tmp = zeros(T, length(solver.vars))
    return Displacement(u, dudx_tmp, solver, global_dofs, 0, maxfevals)
end

function Displacement(solver::AbstractHyperelasticSolver; maxfevals=10^8)
    dim = TopOptProblems.getdim(solver.problem)
    dim == 3 || throw("2D hyperelastic FEA is not supported yet.")
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    F = [zeros(3, 3) for _ in 1:total_ndof/dim]
    dudx_tmp = zeros(T, length(solver.vars))
    return HyperelasticDisplacement(u, F, dudx_tmp, solver, global_dofs, 0, maxfevals)
end

"""
# Arguments
`x` = design variables

# Returns
displacement vector `u`
"""
function (dp::Displacement{T})(x::PseudoDensities) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x.x
    solver()
    return DisplacementResult(copy(solver.u))
end

function (dp::HyperelasticDisplacement{T})(x::PseudoDensities) where {T}
    @unpack solver, global_dofs = dp
    @unpack penalty, problem, xmin = solver
    dp.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x.x
    solver()
    return DisplacementResult(copy(solver.u)) #, copy(solver.F) # I need to add F support
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
function ChainRulesCore.rrule(dp::Displacement, x::PseudoDensities)
    @unpack dudx_tmp, solver, global_dofs = dp
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    # Forward-pass
    # Cholesky factorisation
    u = dp(x)
    return u, Δ -> begin # v
        if hasproperty(Δ, :u)
            solver.rhs .= Δ.u
        else
            solver.rhs .= Δ
        end
        solver(; reuse_fact=true, assemble_f=false)
        dudx_tmp .= 0
        for e in 1:length(x.x)
            _, dρe = get_ρ_dρ(x.x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            Keu = bcmatrix(Kes[e]) * u.u[global_dofs]
            dudx_tmp[e] = -dρe * dot(Keu, solver.lhs[global_dofs])
        end
        return nothing, Tangent{typeof(x)}(; x=dudx_tmp) # J1' * v, J2' * v
    end
end
