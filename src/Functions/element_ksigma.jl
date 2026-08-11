using TopOpt.TopOptProblems: getE
using TopOpt.TrussTopOptProblems: truss_reinit!
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: density, getpenalty

"""
    TrussElementKσ(problem, solver)

Element-wise stress/geometric stiffness matrices for truss domains. Used in
buckling-constrained truss optimization.

Call as `Kσs = Kσsf(u, ρ)` where `u` is the displacement vector and `ρ` is the
penalized/interpolated design. Returns a vector of symmetric matrices.
"""
mutable struct TrussElementKσ{
    T,
    Tp<:TrussProblem,
    TK<:AbstractVector{<:AbstractMatrix{T}},
    TE<:AbstractVector{<:AbstractVector{T}},
    Td<:AbstractVector{<:AbstractMatrix{T}},
    TL<:AbstractVector{T},
    Tg<:AbstractVector{<:Integer},
    Tpen<:AbstractPenalty{T},
} <: AbstractFunction{T}
    problem::Tp
    Kσes::TK
    EALγ_s::TE
    δmat_s::Td
    L_s::TL
    global_dofs::Tg
    penalty::Tpen
    xmin::T
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TrussElementKσ)
    return println(
        io, "TopOpt element stress stiffness matrix (Kσ_e) construction function"
    )
end

function TrussElementKσ(
    problem::TrussProblem{xdim,T}, solver::AbstractFEASolver
) where {xdim,T}
    Es = getE(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    cellvalues = solver.elementinfo.cellvalues
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert ndof_pc == xdim * n_basefuncs "$ndof_pc, $n_basefuncs"
    @assert n_basefuncs == 2

    global_dofs = zeros(Int, ndof_pc)
    δmat = zeros(T, ndof_pc, ndof_pc)
    Kσes = Matrix{T}[]
    EALγ_s = Vector{T}[]
    δmat_s = Matrix{T}[]
    L_s = T[]

    for (cellidx, cell) in enumerate(CellIterator(dh))
        truss_reinit!(cellvalues, cell, As[cellidx])
        celldofs!(global_dofs, dh, cellidx)
        E = Es[cellidx]
        A = As[cellidx]
        L = norm(cell.coords[1] - cell.coords[2])
        R = compute_local_axes(cell.coords[1], cell.coords[2])
        # local axial projection operator (local axis transformation)
        γ = vcat(-R[:, 1], R[:, 1])
        push!(EALγ_s, (E * A / L) * γ)

        fill!(δmat, 0.0)
        for i in 2:size(R, 2)
            δ = vcat(-R[:, i], R[:, i])
            # @assert δ' * γ ≈ 0
            δmat .+= δ * δ'
        end
        push!(δmat_s, copy(δmat))
        push!(L_s, T(L))

        push!(Kσes, zeros(T, ndof_pc, ndof_pc))
    end
    penalty = getpenalty(solver)
    xmin = solver.xmin
    return TrussElementKσ(problem, Kσes, EALγ_s, δmat_s, L_s, global_dofs, penalty, xmin)
end

"""
    (eksig::TrussElementKσ)(u_e::AbstractVector, x_e::Number, ci::Integer)

Compute the stress (geometric) stiffness matrix for **truss element** index
`ci` (axial bar element, no bending or torsion), with nodal deformation `u_e`
and penalized density `x_e`.

The matrix formulation is defined in eqs. (3) and (4) of
[Gavin2014](@cite), and is equivalent to the one used in [Kocvara2002](@cite).

Note: bar axial force is computed using a first-order approximation.
A better approximation would be:
`EA/L * (u3-u1 + 1/(2*L0)*(u4-u2)^2) = EA/L * (γ'*u + 1/2*(δ'*u)^2)`;
see [Gavin2014](@cite).
"""
function (eksig::TrussElementKσ)(u_e::AbstractVector, x_e::Number, ci::Integer)
    @unpack EALγ_s, δmat_s, L_s = eksig
    # x_e scales the cross section
    q_cell = x_e * EALγ_s[ci]' * u_e
    return q_cell / L_s[ci] * δmat_s[ci]
end

function (eksig::TrussElementKσ)(u::DisplacementResult, x::PseudoDensities)
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    @assert getncells(dh.grid) == length(x.x)
    @assert ndofs(dh) == length(u.u)
    for ci in 1:length(x.x)
        celldofs!(global_dofs, dh, ci)
        # Apply the same penalty/interpolation as ElementK and the stiffness assembly
        ρ_e, _ = get_ρ_dρ(x.x[ci], penalty, xmin)
        Kσes[ci] = eksig(u.u[global_dofs], ρ_e, ci)
    end
    return copy(Kσes)
end

function ChainRulesCore.rrule(
    eksig::TrussElementKσ{T}, u::DisplacementResult, x::PseudoDensities
) where {T}
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    Kσes = eksig(u, x)
    function pullback_fn(Δ)
        Δu = zeros(T, size(u.u))
        Δx = zeros(T, size(x.x))
        for ci in 1:length(x.x)
            celldofs!(global_dofs, dh, ci)
            # Jacobian with respect to (u_e, ρ_e)
            function vec_eksig_fn(uρ_vec)
                u_e = uρ_vec[1:(end - 1)]
                ρ_e = uρ_vec[end]
                return vec(eksig(u_e, ρ_e, ci))
            end
            ρ_e, dρ_dx = get_ρ_dρ(x.x[ci], penalty, xmin)
            jac_cell = ForwardDiff.jacobian(vec_eksig_fn, [u.u[global_dofs]; ρ_e])
            jtv = jac_cell' * vec(Δ[ci])
            Δu[global_dofs] += jtv[1:(end - 1)]
            # Chain rule: d(Kσ)/d(x) = d(Kσ)/d(ρ) * d(ρ)/d(x)
            Δx[ci] = jtv[end] * dρ_dx
        end
        return Tangent{typeof(eksig)}(;
            problem=NoTangent(),
            Kσes=Δ,
            EALγ_s=NoTangent(),
            δmat_s=NoTangent(),
            L_s=NoTangent(),
            global_dofs=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
        ),
        Tangent{typeof(u)}(; u=Δu),
        Tangent{typeof(x)}(; x=Δx)
    end
    return Kσes, pullback_fn
end

#####################################

# TODO ElementKσ for volumetic cells
