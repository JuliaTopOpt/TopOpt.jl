"""
    TrussStress(problem, solver)

Element-wise macroscopic axial stress for truss problems. Computes the axial
stress in each truss member from nodal displacements and the penalized design.
Call as `σ = σf(u, ρ)` where `u` is the displacement vector and `ρ` is the
penalized/interpolated design.

The axial stress is computed as `σ_e = -(R_e · Ke_e · u_e)[1] / A_e`, where
`R_e` is the local-to-global transformation matrix, `Ke_e` is the element
stiffness matrix, and `A_e` is the cross-sectional area. Compressive stress
is negative, tensile stress is positive. See [Gavin2014](@cite) for the
truss finite element formulation.
"""
mutable struct TrussStress{
    T,Ts<:AbstractVector{T},Tu<:Displacement,Tt<:AbstractVector{<:AbstractMatrix{T}}
} <: AbstractFunction{T}
    σ::Ts # stress vector, axial stress per cell
    u_fn::Tu
    transf_matrices::Tt
    fevals::Int
    maxfevals::Int
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TrussStress)
    return println(io, "TopOpt truss stress function")
end

"""
    TrussStress(solver; maxfevals=10^8)

Construct the TrussStress function struct.
"""
function TrussStress(solver::AbstractFEASolver; maxfevals=10^8)
    # TrussStress is only valid for truss problems
    @assert solver.problem isa TrussProblem "TrussStress can only be used with TrussProblem. Got $(typeof(solver.problem))"
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

"""
    rrule for TrussStress

Adjoint-based differentiation. The stress σ_e = -(R_e · Ke_e · u_e)[1] / A_e
depends on x through both the penalized stiffness Ke_e = ρ(x)_e · Ke_0_e and the
displacement u = K⁻¹f. The pullback solves one adjoint system per evaluation.
"""
function ChainRulesCore.rrule(ts::TrussStress{T}, x::PseudoDensities) where {T}
    @unpack σ, transf_matrices, u_fn = ts
    @unpack global_dofs, solver = u_fn
    @unpack penalty, problem, xmin = solver
    dh = getdh(problem)
    As = getA(problem)
    @unpack Kes = solver.elementinfo

    # Forward pass
    σ_val = ts(x)
    u = u_fn(x)  # DisplacementResult — factorization is now cached in solver

    function pullback_fn(Δ)
        Δx = zeros(T, length(x.x))

        # Build the adjoint RHS from dσ_e/du for all elements.
        # σ_e = -(R_e · Ke_e · u_e)[1] / A_e
        # dσ_e/du[global_dofs] = -(R_e · Ke_e)[1, :] / A_e
        adj_rhs = zeros(T, length(u.u))
        for e in 1:length(x.x)
            celldofs!(global_dofs, dh, e)
            Ke = rawmatrix(Kes[e])
            grad_u = (transf_matrices[e] * Ke)[1, :] / As[e]
            adj_rhs[global_dofs] .-= Δ[e] .* grad_u
        end

        # Solve the adjoint system K * λ = adj_rhs (reuse factorization)
        solver.rhs .= adj_rhs
        solver(; reuse_fact=true, assemble_f=false)
        λ = solver.lhs

        # Gradient assembly:
        # dσ_e/dx_j has two parts:
        #   1. Direct (only for e=j): -(R_e · Ke_0_e · u_e)[1] / A_e * dρ/dx
        #   2. Through u: Σ_e Δ_e * (dσ_e/du · du/dx_j)
        #      du/dx_j = -K⁻¹ · dρ/dx · Ke_0_j · u
        #      => adjoint term = -dρ/dx * dot(Ke_0_j · u_j, λ_j)
        for j in 1:length(x.x)
            _, dρ_dx = get_ρ_dρ(x.x[j], penalty, xmin)
            celldofs!(global_dofs, dh, j)
            Ke_0_u = rawmatrix(Kes[j]) * u.u[global_dofs]
            adjoint = -dρ_dx * dot(Ke_0_u, λ[global_dofs])
            Δx[j] = adjoint
        end

        return NoTangent(), Tangent{typeof(x)}(; x=Δx)
    end

    return σ_val, pullback_fn
end
