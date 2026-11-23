"""
    BucklingConeFun(solver::AbstractFEASolver, λ̄)

Differentiable buckling-cone matrix `P = K + λ̄ Kσ`: the linearized buckling
constraint `λ₁ ≥ λ̄` on the fundamental Buckling Load Factor is equivalent to
`P ⪰ 0`, where `K` is the (penalized, condensed) elastic stiffness and `Kσ`
the geometric stiffness assembled from the prebuckling displacement field
(see [`TopOptProblems.buckling`](@ref) and [Bathe1996](@cite) §9.3).

Call as `P = buck(PseudoDensities(x))`; the result is a sparse symmetric
matrix. This function is differentiable: its reverse rule composes the
existing rrules — `ElementKFun`/`AssembleKFun` for the `K` part,
[`ContinuumElementKσFun`](@ref) for the `Kσ` part (including the stress
dependence on the displacement field), and `DisplacementFun` for the
adjoint of that `u(x)` dependence. Compositions like `v'Pv`, `λmin(P)` or
`NonconvexCore` `PSDCone` constraints can therefore be differentiated with
respect to the design variables, enabling buckling-constrained topology
optimization (e.g. via cutting-plane or projected augmented Lagrangian
methods).

Only valid for `StiffnessTopOptProblem` with homogeneous Dirichlet BCs (the
`DisplacementFun` adjoint assumes the standard condensed solve).
"""
mutable struct BucklingConeFun{
    T,TS<:AbstractFEASolver,TE,TA,TEK,TDP
} <: AbstractFunction{T}
    solver::TS
    λ̄::T
    eksig::TE          # ContinuumElementKσFun
    akf::TA            # AssembleKFun
    ekf::TEK           # ElementKFun
    dpf::TDP           # DisplacementFun
end

Utilities.getpenalty(bc::BucklingConeFun) = getpenalty(bc.solver)

function BucklingConeFun(solver::AbstractFEASolver, λ̄)
    problem = solver.problem
    problem isa StiffnessTopOptProblem || throw(
        ArgumentError(
            "BucklingConeFun can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(problem))",
        ),
    )
    # All the sub-functions are constructed once: several of them allocate
    # workspace or (re)initialize the solver state in their constructors, so
    # they must not be rebuilt inside the pullback.
    eksig = ContinuumElementKσFun(problem, solver)
    akf = AssembleKFun(problem)
    ekf = ElementKFun(solver)
    dpf = DisplacementFun(solver)
    return BucklingConeFun(solver, λ̄, eksig, akf, ekf, dpf)
end

function (bc::BucklingConeFun)(x::PseudoDensities)
    @unpack solver, λ̄ = bc
    problem = solver.problem
    solver.vars .= x.x
    solver()
    K, Kσ = buckling(
        problem, solver.globalinfo, solver.elementinfo, solver.vars,
        getpenalty(solver), solver.xmin,
    )
    Kmat = K isa Symmetric ? K.data : K
    Kσmat = Kσ isa Symmetric ? Kσ.data : Kσ
    # Wrap in `Symmetric`: Ferrite's condensed matrices store both triangles,
    # with the prescribed-COLUMN entries retaining assembly values (they
    # multiply u_p = 0 and are ignored by the wrapper, which reads only the
    # upper triangle — where the apply!'d prescribed rows are identity).
    # This also sidesteps 1-ulp value asymmetries in the stored Kσ triangles
    # (different summation orders in G'ψG), which trip `issymmetric`'s exact
    # comparison on the raw data.
    return Symmetric(Kmat + λ̄ * Kσmat, :U)
end

function ChainRulesCore.rrule(bc::BucklingConeFun{T}, x::PseudoDensities) where {T}
    @unpack solver, λ̄, eksig, akf, ekf, dpf = bc
    problem = solver.problem
    dh = problem.ch.dh
    penalty = getpenalty(solver)
    xmin = solver.xmin
    P = bc(x)
    function buck_pullback(ΔP)
        ΔP = ChainRulesCore.unthunk(ΔP)
        if ΔP isa ChainRulesCore.AbstractZero
            return (
                Tangent{typeof(bc)}(; solver=NoTangent(), λ̄=NoTangent()),
                Tangent{typeof(x)}(; x=zeros(T, length(x.x))),
            )
        end
        Δx = zeros(T, length(x.x))
        # The forward returns Symmetric(Ptri, :U): the wrapper reads the
        # upper triangle only, so the stored LOWER entries have zero
        # cotangent, and the upper off-diagonals carry ΔP[i,j] + ΔP[j,i].
        # Prescribed-DOF entries: the off-diagonal rows/cols are constant
        # zeros (dropped), while the DIAGONAL is Ferrite's apply! value
        # m = meandiag(K_raw) — the mean of the fully assembled diagonal —
        # a global x-dependent quantity whose gradient flows through
        # trace(K0e)/n. Kσ's prescribed diagonal is exactly 0 (see
        # `buckling`), so that one is dropped.
        ΔPs = (ΔP + ΔP') / 2
        ΔKs = triu(ΔPs + ΔPs' - diagm(0 => diag(ΔPs)))
        ΔKσs = λ̄ .* ΔKs
        s_diag = 0.0
        for d in problem.ch.prescribed_dofs
            s_diag += ΔPs[d, d]
            ΔKs[d, :] .= 0.0
            ΔKs[:, d] .= 0.0
            ΔKσs[d, :] .= 0.0
            ΔKσs[:, d] .= 0.0
        end
        if !iszero(s_diag)
            ndof_tot = ndofs(dh)
            for e in eachindex(x.x)
                _, drho = get_ρ_dρ(x.x[e], penalty, xmin)
                Δx[e] += s_diag * drho * sum(diag(ekf.Kes_0[e])) / ndof_tot
            end
        end

        # ---- K part: AssembleKFun reverse -> ElementKFun reverse ----
        _, ak_pb = ChainRulesCore.rrule(akf, ekf(x))
        ΔKes = ak_pb(ΔKs)[2]
        _, ek_pb = ChainRulesCore.rrule(ekf, x)
        Δx_K = ek_pb(ΔKes)[2].x
        Δx .+= Δx_K

        # ---- Kσ part: assembly scatter reverse -> ContinuumElementKσFun
        # reverse (u and x parts) -> DisplacementFun adjoint for the
        # u-dependence. The prescribed-DOF entries of Kσ are constant zeros
        # (see `buckling`), so their cotangents must not flow back.
        u = solver.u
        global_dofs = zeros(Int, ndofs_per_cell(dh))
        ΔKσes = [zeros(T, ndofs_per_cell(dh), ndofs_per_cell(dh)) for _ in 1:getncells(dh.grid)]
        for (ci, _) in enumerate(CellIterator(dh))
            celldofs!(global_dofs, dh, ci)
            ΔKσes[ci] = ΔKσs[global_dofs, global_dofs]
        end
        _, eksig_pb = ChainRulesCore.rrule(eksig, DisplacementResult(copy(u)), x)
        _, Δu_t, Δx_t = eksig_pb(ΔKσes)
        Δx .+= Δx_t.x
        # Δu through the displacement adjoint (reuses the cached factor)
        _, dp_pb = ChainRulesCore.rrule(dpf, x)
        Δx_u = dp_pb(Δu_t.u)[2].x
        Δx .+= Δx_u

        return (
            Tangent{typeof(bc)}(; solver=NoTangent(), λ̄=NoTangent()),
            Tangent{typeof(x)}(; x=Δx),
        )
    end
    return P, buck_pullback
end
