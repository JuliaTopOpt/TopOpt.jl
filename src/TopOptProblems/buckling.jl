using Einsum
using ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using ..Utilities: density
import ..Utilities: get_ρ_dρ

"""
    get_Kσs(sp, u_dofs, cellvalues, vars, penalty, xmin)

Compute the per-cell geometric stiffness matrices `Kσ` from the displacement
DOFs `u_dofs` and the `cellvalues` of a stiffness topology optimization
problem `sp`. Each element's Kσ is scaled by the penalized density
`ρ = penalty(density(vars[e], xmin))` (or `density(penalty(vars[e]), xmin)`
depending on `PENALTY_BEFORE_INTERPOLATION`), consistent with the stiffness
assembly. Used by `buckling`.

The stress is computed using the 3D isotropic constitutive law (plane
strain for 2D problems), with Lamé constants
`λ = Eν/((1+ν)(1-2ν))` and `2μ = E/(1+ν)`, matching the stiffness matrix
assembly. See [Bathe1996](@cite) §4.2 for the plane-strain formulation
and [CookMalkusPlesha1989](@cite) §13 for the geometric stiffness matrix.
"""
function get_Kσs(
    sp::StiffnessTopOptProblem{xdim,TT},
    u_dofs,
    cellvalues,
    vars=ones(TT, getncells(sp.ch.dh.grid)),
    penalty=PowerPenaltyFun{TT}(1),
    xmin=TT(1//1000),
) where {xdim,TT}
    E = getE(sp)
    ν = getν(sp)
    dh = sp.ch.dh
    # usually ndof_pc = xdim * n_basefuncs
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndof_pc)
    Kσs = [zeros(TT, ndof_pc, ndof_pc) for i in 1:getncells(dh.grid)]
    Kσ_e = zeros(TT, ndof_pc, ndof_pc)
    # block-diagonal - block σ_e = σ_ij, i,j in xdim
    ψ_e = zeros(TT, xdim * xdim, xdim * xdim)
    G = zeros(TT, xdim * xdim, xdim * n_basefuncs)
    δ = Matrix(TT(1.0)I, xdim, xdim)
    ϵ = zeros(TT, xdim, xdim)
    σ = zeros(TT, xdim, xdim)
    # u_i,j: partial derivative
    u_p = zeros(TT, xdim, xdim)
    # 3D isotropic constitutive law (plane strain), consistent with the
    # stiffness matrix assembly in matrices_and_vectors.jl:
    #   σ_ij = λ * ε_kk * δ_ij + 2μ * ε_ij
    #   λ = E*ν/((1+ν)*(1-2ν)),  2μ = E/(1+ν)
    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    two_mu = E / (1 + ν)
    for (cellidx, cell) in enumerate(CellIterator(dh))
        Kσ_e .= 0
        reinit!(cellvalues, cell)
        # get cell's dof's global dof indices, i.e. CC_a^e
        celldofs!(global_dofs, dh, cellidx)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for d in 1:xdim
                ψ_e[((d - 1) * xdim + 1):(d * xdim), ((d - 1) * xdim + 1):(d * xdim)] .= 0
            end
            for a in 1:n_basefuncs
                ∇ϕ = shape_gradient(cellvalues, q_point, a)
                _u = @view u_dofs[(@view global_dofs[xdim * (a - 1) .+ (1:xdim)])]
                # u_i,j, i for spatial xdim, j for partial derivative
                @einsum u_p[i, j] = _u[i] * ∇ϕ[j]
                # effect of the quadratic term in the strain formula have on the stress field is ignored
                @einsum ϵ[i, j] = 1 / 2 * (u_p[i, j] + u_p[j, i])
                # isotropic solid — 3D (plane strain) constitutive law
                @einsum σ[i, j] = λ * δ[i, j] * ϵ[k, k] + two_mu * ϵ[i, j]
                for d in 1:xdim
                    # block diagonal
                    ψ_e[
                        ((d - 1) * xdim .+ 1):(d * xdim), ((d - 1) * xdim .+ 1):(d * xdim)
                    ] .+= σ
                    G[(xdim * (d - 1) + 1):(xdim * d), (a - 1) * xdim + d] .= ∇ϕ
                end
            end
            Kσ_e .+= G' * ψ_e * G * dΩ
        end
        # Scale by the penalized element density, consistent with stiffness assembly
        ρ_e, _ = get_ρ_dρ(vars[cellidx], penalty, xmin)
        Kσs[cellidx] .= ρ_e * Kσ_e
    end

    return Kσs
end

"""
    buckling(problem, ginfo, einfo, [vars, penalty, xmin])

Assemble the global geometric stiffness matrix `Kσ` for a stiffness topology
optimization `problem` from the global FEA info `ginfo` and element FEA info
`einfo`. Returns `(K, Kσ)`, the elastic and geometric stiffness matrices. Each
element's geometric stiffness is scaled by the penalized density `ρ(vars[e])`,
consistent with the stiffness assembly in `assemble!`.

See [Bathe1996](@cite) §9.3 for linearized buckling analysis and
[CookMalkusPlesha1989](@cite) §13 for the geometric stiffness assembly.
"""
function buckling(
    problem::StiffnessTopOptProblem{xdim,T},
    ginfo,
    einfo,
    vars=ones(T, getncells(problem.ch.dh.grid)),
    penalty=PowerPenaltyFun{T}(1),
    xmin=T(1) / 1000,
) where {xdim,T}
    dh = problem.ch.dh

    u = ginfo.K \ ginfo.f
    Kσs = get_Kσs(problem, u, einfo.cellvalues, vars, penalty, xmin)
    Kσ = deepcopy(ginfo.K)

    if Kσ isa Symmetric
        Kσ.data.nzval .= 0
        # Kσ is zeroed explicitly above; disable fillzero to preserve that.
        assembler = Ferrite.start_assemble(Kσ.data; fillzero=false)
    else
        Kσ.nzval .= 0
        assembler = Ferrite.start_assemble(Kσ; fillzero=false)
    end

    # * assemble global geometric stiffness matrix
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    Kσ_e = zeros(T, size(Kσs[1]))
    _celliterator = CellIterator(dh)
    TK = eltype(Kσs)
    for (i, cell) in enumerate(_celliterator)
        celldofs!(global_dofs, dh, i)
        if TK <: Symmetric
            Ferrite.assemble!(assembler, global_dofs, Kσs[i].data)
        else
            Ferrite.assemble!(assembler, global_dofs, Kσs[i])
        end
    end

    return ginfo.K, Kσ
end
