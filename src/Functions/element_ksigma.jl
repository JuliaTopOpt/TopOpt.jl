using TopOpt.TopOptProblems: getE
import TopOpt.TopOptProblems: element_kσ
using TopOpt.TrussTopOptProblems: truss_reinit!
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: density, getpenalty

"""
    TrussElementKσFun(problem, solver)

Element-wise stress/geometric stiffness matrices for truss domains. Used in
buckling-constrained truss optimization.

Call as `Kσs = Kσsf(u, ρ)` where `u` is the displacement vector and `ρ` is the
penalized/interpolated design. Returns a vector of symmetric matrices.
"""
mutable struct TrussElementKσFun{
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

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TrussElementKσFun)
    return println(
        io, "TopOpt element stress stiffness matrix (Kσ_e) construction function"
    )
end

function TrussElementKσFun(
    problem::TrussProblem{xdim,T}, solver::AbstractFEASolver
) where {xdim,T}
    Es = getE(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    cellvalues = solver.elementinfo.cellvalues
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    ndof_pc == xdim * n_basefuncs || throw(
        DimensionMismatch(
            "TrussElementKσFun: expected ndof_pc=$(xdim * n_basefuncs), got $ndof_pc"
        ),
    )
    n_basefuncs == 2 || throw(
        ArgumentError(
            "TrussElementKσFun: expected 2 base functions for truss element, got $n_basefuncs",
        ),
    )

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
    return TrussElementKσFun(problem, Kσes, EALγ_s, δmat_s, L_s, global_dofs, penalty, xmin)
end

"""
    (eksig::TrussElementKσFun)(u_e::AbstractVector, x_e::Number, ci::Integer)

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
function (eksig::TrussElementKσFun)(u_e::AbstractVector, x_e::Number, ci::Integer)
    @unpack EALγ_s, δmat_s, L_s = eksig
    # x_e scales the cross section
    q_cell = x_e * EALγ_s[ci]' * u_e
    return q_cell / L_s[ci] * δmat_s[ci]
end

function (eksig::TrussElementKσFun)(u::DisplacementResult, x::PseudoDensities)
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    ncells = getncells(dh.grid)
    ncells == length(x.x) || throw(
        DimensionMismatch(
            "TrussElementKσFun: expected $(ncells) cells, got $(length(x.x))"
        ),
    )
    ndofs(dh) == length(u.u) || throw(
        DimensionMismatch(
            "TrussElementKσFun: expected $(ndofs(dh)) DOFs, got $(length(u.u))"
        ),
    )
    for ci in eachindex(x.x)
        celldofs!(global_dofs, dh, ci)
        # Apply the same penalty/interpolation as ElementKFun and the stiffness assembly
        ρ_e, _ = get_ρ_dρ(x.x[ci], penalty, xmin)
        Kσes[ci] = eksig(u.u[global_dofs], ρ_e, ci)
    end
    return copy(Kσes)
end

function ChainRulesCore.rrule(
    eksig::TrussElementKσFun{T}, u::DisplacementResult, x::PseudoDensities
) where {T}
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    Kσes = eksig(u, x)
    function pullback_fn(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        Δu = zeros(T, size(u.u))
        Δx = zeros(T, size(x.x))
        for ci in eachindex(x.x)
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

"""
    ContinuumElementKσFun(problem, solver)

Element-wise geometric stiffness matrices for continuum (2D/3D) domains,
computed from the nodal displacements with the plane-strain (3D isotropic)
constitutive law of the stiffness assembly. Used in buckling-constrained
continuum optimization, e.g. as a differentiable penalty on the lowest
buckling eigenvalue.

Call as `Kσs = Kσsf(u, x)` where `u` is the displacement vector (a
[`DisplacementResult`](@ref)) and `x` is the penalized/interpolated design.
Returns a vector of symmetric matrices, one per element, each scaled by the
penalized density `ρ_e = penalty(density(x_e, xmin))`, consistent with the
stiffness assembly and [`TopOptProblems.buckling`](@ref).
"""
mutable struct ContinuumElementKσFun{
    T,
    Tp<:StiffnessTopOptProblem,
    TK<:AbstractVector{<:AbstractMatrix{T}},
    Tcv,
    Tcc,
    Tg<:AbstractVector{<:Integer},
    Tpen<:AbstractPenalty{T},
} <: AbstractFunction{T}
    problem::Tp
    Kσes::TK
    cellvalues::Tcv
    cellcache::Tcc
    global_dofs::Tg
    penalty::Tpen
    xmin::T
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::ContinuumElementKσFun)
    return println(
        io, "TopOpt continuum element stress stiffness matrix (Kσ_e) construction function"
    )
end

function ContinuumElementKσFun(
    problem::StiffnessTopOptProblem{xdim,T}, solver::AbstractFEASolver
) where {xdim,T}
    dh = problem.ch.dh
    cellvalues = solver.elementinfo.cellvalues
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    ndof_pc == xdim * n_basefuncs || throw(
        DimensionMismatch(
            "ContinuumElementKσFun: expected ndof_pc=$(xdim * n_basefuncs), got $ndof_pc",
        ),
    )
    global_dofs = zeros(Int, ndof_pc)
    Kσes = [zeros(T, ndof_pc, ndof_pc) for _ in 1:getncells(dh.grid)]
    cellcache = Ferrite.CellCache(dh)
    penalty = getpenalty(solver)
    xmin = solver.xmin
    return ContinuumElementKσFun(
        problem, Kσes, cellvalues, cellcache, global_dofs, penalty, xmin
    )
end

"""
    (eksig::ContinuumElementKσFun)(u_e::AbstractVector, x_e::Number, ci::Integer)

Compute the geometric stiffness matrix for continuum element index `ci` from
its nodal displacements `u_e` and penalized density `x_e`.
"""
function (eksig::ContinuumElementKσFun)(u_e::AbstractVector, x_e::Number, ci::Integer)
    Ferrite.reinit!(eksig.cellcache, ci)
    reinit!(eksig.cellvalues, eksig.cellcache)
    return x_e * element_kσ(eksig.problem, eksig.cellvalues, u_e)
end

function (eksig::ContinuumElementKσFun)(u::DisplacementResult, x::PseudoDensities)
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    ncells = getncells(dh.grid)
    ncells == length(x.x) || throw(
        DimensionMismatch(
            "ContinuumElementKσFun: expected $(ncells) cells, got $(length(x.x))"
        ),
    )
    ndofs(dh) == length(u.u) || throw(
        DimensionMismatch(
            "ContinuumElementKσFun: expected $(ndofs(dh)) DOFs, got $(length(u.u))"
        ),
    )
    for ci in eachindex(x.x)
        celldofs!(global_dofs, dh, ci)
        # Apply the same penalty/interpolation as ElementKFun and the stiffness assembly
        ρ_e, _ = get_ρ_dρ(x.x[ci], penalty, xmin)
        Kσes[ci] = eksig(u.u[global_dofs], ρ_e, ci)
    end
    return copy(Kσes)
end

function ChainRulesCore.rrule(
    eksig::ContinuumElementKσFun{T}, u::DisplacementResult, x::PseudoDensities
) where {T}
    @unpack problem, Kσes, global_dofs, penalty, xmin = eksig
    dh = problem.ch.dh
    Kσes = eksig(u, x)
    function pullback_fn(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        Δu = zeros(T, size(u.u))
        Δx = zeros(T, size(x.x))
        for ci in eachindex(x.x)
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
            cellvalues=NoTangent(),
            cellcache=NoTangent(),
            global_dofs=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
        ),
        Tangent{typeof(u)}(; u=Δu),
        Tangent{typeof(x)}(; x=Δx)
    end
    return Kσes, pullback_fn
end
