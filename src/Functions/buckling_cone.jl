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

"""
    BucklingCutFun(solver::AbstractFEASolver, λ̄, v)

A scalar buckling cut `vᵀ(K + λ̄Kσ)v` for the cutting-plane enforcement of
the constraint λ₁ ≥ λ̄: `v` is a (fixed) buckling mode of the current
design, and the cut value `vᵀ(K + λ̄Kσ)v ≥ 0` is a classical inequality
constraint. The gradient is the closed-form sensitivity with `v` held
fixed (see `_cut_gradient`), so it is O(m) memory at any mesh size; no
global `K + λ̄Kσ` matrix is ever formed.

Multiple call methods, mirroring [`ContinuumElementKσFun`](@ref); each
delegates to the next so the value and gradient logic live in one place:

- `cut(x̂::PseudoDensities)` — self-contained: assembles and solves at `x̂`,
  then evaluates the cut at the resulting displacement field. The gradient
  includes the design dependence of that field (the adjoint term
  `−λ̄ w_eᵀK0e u_e`).
- `cut(x̂::PseudoDensities, u::AbstractVector)` — reuses a *precomputed*
  displacement field; the per-cell stress state is computed from `u`
  without re-solving. The solver's `K` and element state must be assembled
  at the same `x̂`. Since `u` is a constant input here, the gradient is the
  *frozen-state* gradient (only the explicit design terms; no adjoint
  solve) — when the caller obtained `u` from the design, capture that
  dependence through the self-contained form (e.g. the
  [`BucklingCutsFun`](@ref) bundle) instead.
- `cut(x̂::PseudoDensities, u, Kσ0s::AbstractVector)` — reuses the
  displacements *and* the per-cell unpenalized geometric stiffness
  matrices `Kσ0s` (as returned by [`element_kσ`](@ref)), e.g. when they
  were already computed for another constraint. No solve and no stress
  assembly happen in this form; same frozen-state gradient semantics as
  the `u` form.

The precomputed-input forms exist so several constraints evaluated at the
same design can share one displacement solve and one stress assembly —
compose them in a set constraint (see [`BucklingCutsFun`](@ref)) or a
`SetProduct` closure. The buckling modes must vanish on the prescribed
DOFs (they are modes of the condensed pencil), which makes the per-cell
quadratic form of the value exact.
"""
struct BucklingCutFun{TS<:AbstractFEASolver,T,TV<:AbstractVector{T}} <: AbstractFunction{T}
    solver::TS
    λ̄::T
    v::TV
end

# The per-cell unpenalized geometric stiffness matrices (the stress state)
# of a *given* displacement field. Shared by the precomputed-input call
# forms of BucklingCutFun and by BucklingCutsFun.
function _element_kσs(solver::AbstractFEASolver, u::AbstractVector)
    problem = solver.problem
    cellvalues = solver.elementinfo.cellvalues
    dh = problem.ch.dh
    cv_cell = Ferrite.CellCache(dh)
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    m = getncells(dh.grid)
    Kσ0s = Vector{Matrix{eltype(u)}}(undef, m)
    for ci in 1:m
        celldofs!(global_dofs, dh, ci)
        Ferrite.reinit!(cv_cell, ci)
        reinit!(cellvalues, cv_cell)
        Kσ0s[ci] = element_kσ(problem, cellvalues, u[global_dofs])
    end
    return Kσ0s
end

# The shared closed-form gradient:
#   ∂/∂x̂_e [vᵀ(K + λ̄Kσ)v] = dρ/dx̂_e · [v_eᵀK0e v_e + λ̄ v_eᵀG0e v_e − λ̄ w_eᵀK0e u_e],
# where w = K⁻¹ load solves the adjoint problem with the per-cell load
# assembled through the analytic `element_kσ` reverse (cotangent ρ_e·v_e v_eᵀ,
# the unpenalized stress's linear dependence on u_e). `Kσ0s === nothing`
# means the stress is recomputed from `u`; otherwise the given per-cell
# matrices are reused.
#
# With `u_dep = false` (the precomputed-input call forms) the displacement
# is a *constant* input: there is no implicit x̂-dependence through the
# solve, so the adjoint term `−λ̄ w_eᵀK0e u_e` is dropped and only the
# explicit-density terms remain. With `u_dep = true` (the self-contained
# forms) `u = u(x̂)` and the full three-term chain rule applies.
function _cut_gradient(
    cut::BucklingCutFun, x̂::PseudoDensities, u::AbstractVector, Kσ0s;
    u_dep::Bool=false,
)
    solver = cut.solver
    problem = solver.problem
    penalty = getpenalty(solver)
    xmin = solver.xmin
    cellvalues = solver.elementinfo.cellvalues
    dh = problem.ch.dh
    cv_cell = Ferrite.CellCache(dh)
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    m = getncells(dh.grid)
    grad = zeros(eltype(x̂.x), m)
    if u_dep
        # the adjoint load from vᵀ(∇u Kσ)v, per cell via the analytic
        # reverse, then the adjoint solve w = K⁻¹ load (the cached
        # factorization is reused)
        load = zeros(size(u))
        for ci in 1:m
            celldofs!(global_dofs, dh, ci)
            Ferrite.reinit!(cv_cell, ci)
            reinit!(cellvalues, cv_cell)
            ve = cut.v[global_dofs]
            ρ_e, _ = get_ρ_dρ(x̂.x[ci], penalty, xmin)
            ΔKe = ρ_e .* (ve * ve')
            _, _, _, Δue = ChainRulesCore.rrule(
                element_kσ, problem, cellvalues, u[global_dofs]
            )[2](ΔKe)
            load[global_dofs] .+= Δue
        end
        solver.rhs .= load
        solver(; reuse_fact=true, assemble_f=false)
        w = solver.lhs
    end
    for ci in 1:m
        celldofs!(global_dofs, dh, ci)
        Ferrite.reinit!(cv_cell, ci)
        reinit!(cellvalues, cv_cell)
        ve = cut.v[global_dofs]
        ue = u[global_dofs]
        K0e = Matrix(rawmatrix(solver.elementinfo.Kes[ci]))
        G0e = Kσ0s === nothing ?
              element_kσ(problem, cellvalues, ue) : Kσ0s[ci]
        _, dρ = get_ρ_dρ(x̂.x[ci], penalty, xmin)
        grad[ci] = dρ * (
            dot(ve, K0e, ve) + cut.λ̄ * dot(ve, G0e, ve) -
            (u_dep ? cut.λ̄ * dot(w[global_dofs], K0e, ue) : zero(eltype(x̂.x)))
        )
    end
    return grad
end

# ---- self-contained form ----
function (cut::BucklingCutFun)(x̂::PseudoDensities)
    solver = cut.solver
    solver.vars .= x̂.x
    solver()
    return cut(x̂, solver.u)
end
function ChainRulesCore.rrule(cut::BucklingCutFun, x̂::PseudoDensities)
    val = cut(x̂)
    function cut_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        grad = _cut_gradient(cut, x̂, cut.solver.u, nothing; u_dep=true)
        return Tangent{typeof(cut)}(;
            solver=NoTangent(), λ̄=NoTangent(), v=NoTangent()
        ),
        Tangent{typeof(x̂)}(; x=grad .* Δ)
    end
    return val, cut_pb
end

# ---- precomputed-displacement form ----
function (cut::BucklingCutFun)(x̂::PseudoDensities, u::AbstractVector)
    return cut(x̂, u, _element_kσs(cut.solver, u))
end
function ChainRulesCore.rrule(
    cut::BucklingCutFun, x̂::PseudoDensities, u::AbstractVector
)
    Kσ0s = _element_kσs(cut.solver, u)
    val = cut(x̂, u, Kσ0s)
    function cut_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        grad = _cut_gradient(cut, x̂, u, Kσ0s; u_dep=false)
        return Tangent{typeof(cut)}(;
            solver=NoTangent(), λ̄=NoTangent(), v=NoTangent()
        ),
        Tangent{typeof(x̂)}(; x=grad .* Δ),
        NoTangent()
    end
    return val, cut_pb
end

# ---- precomputed displacements and stress (per-cell Kσ0) form ----
function (cut::BucklingCutFun)(
    x̂::PseudoDensities, u::AbstractVector, Kσ0s::AbstractVector
)
    solver = cut.solver
    penalty = getpenalty(solver)
    xmin = solver.xmin
    dh = solver.problem.ch.dh
    K = solver.globalinfo.K
    val = dot(cut.v, K * cut.v)
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    for ci in eachindex(Kσ0s)
        celldofs!(global_dofs, dh, ci)
        ve = cut.v[global_dofs]
        ρ_e, _ = get_ρ_dρ(x̂.x[ci], penalty, xmin)
        val += cut.λ̄ * ρ_e * dot(ve, Kσ0s[ci], ve)
    end
    return val
end
function ChainRulesCore.rrule(
    cut::BucklingCutFun, x̂::PseudoDensities, u::AbstractVector,
    Kσ0s::AbstractVector,
)
    val = cut(x̂, u, Kσ0s)
    function cut_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        grad = _cut_gradient(cut, x̂, u, Kσ0s; u_dep=false)
        return Tangent{typeof(cut)}(;
            solver=NoTangent(), λ̄=NoTangent(), v=NoTangent()
        ),
        Tangent{typeof(x̂)}(; x=grad .* Δ),
        NoTangent(),
        NoTangent()
    end
    return val, cut_pb
end

"""
    BucklingCutsFun(solver::AbstractFEASolver, λ̄, V)

The vector-valued buckling cuts `[vᵢᵀ(K + λ̄Kσ)vᵢ]` for a list of fixed
buckling modes `V = [v₁, v₂, …]`. One evaluation solves the prebuckling
displacement field and assembles the per-cell stress state **once**, then
evaluates every cut from that shared state through the precomputed-input
call forms of [`BucklingCutFun`](@ref); the reverse rule assembles the
Jacobian row by row (one adjoint solve per mode, all reusing the same
cached factorization).

Intended for the set-constraint form: enforce all cuts jointly as

```julia
add_set_constraint!(model, BucklingCutsFun(solver, λ̄, V), Nonnegatives(length(V)))
```

which NonconvexCore records as one function-in-set constraint and
NonconvexAugLagLab evaluates as one bundle, so the shared solve and stress
assembly run once per evaluation instead of once per constraint.

The same call forms as [`BucklingCutFun`](@ref) are available:
`cuts(x)` (self-contained, `x` the design vector), `cuts(x̂, u)` and
`cuts(x̂, u, Kσ0s)` (reusing precomputed state; the solver's `K` and
element state must be assembled at `x̂`, and the gradients are the
frozen-state ones).
"""
struct BucklingCutsFun{TS<:AbstractFEASolver,T,TV,TC}
    solver::TS
    λ̄::T
    V::TV
    cuts::TC
end
function BucklingCutsFun(solver::AbstractFEASolver, λ̄, V)
    cuts = [BucklingCutFun(solver, λ̄, v) for v in V]
    return BucklingCutsFun(solver, λ̄, V, cuts)
end

function (cuts::BucklingCutsFun)(x̂::PseudoDensities)
    solver = cuts.solver
    solver.vars .= x̂.x
    solver()
    return cuts(x̂, solver.u)
end
function (cuts::BucklingCutsFun)(x::AbstractVector)
    return cuts(PseudoDensities(x))
end

function (cuts::BucklingCutsFun)(x̂::PseudoDensities, u::AbstractVector)
    return cuts(x̂, u, _element_kσs(cuts.solver, u))
end

function (cuts::BucklingCutsFun)(
    x̂::PseudoDensities, u::AbstractVector, Kσ0s::AbstractVector
)
    return [cut(x̂, u, Kσ0s) for cut in cuts.cuts]
end

# The shared rrule: one solve and one stress assembly in the forward, and
# one adjoint solve per mode in the pullback (rows of the Jacobian) for the
# self-contained forms (`u_dep = true`); the precomputed-input forms are
# frozen-state gradients (`u_dep = false`).
function _cuts_rrule_pullback(
    cuts::BucklingCutsFun, x̂::PseudoDensities, u, Kσ0s, Δ;
    u_dep::Bool=false,
)
    Δ = ChainRulesCore.unthunk(Δ)
    grad = zeros(eltype(x̂.x), length(x̂.x))
    if !(Δ isa ChainRulesCore.AbstractZero)
        Δv = Δ isa AbstractVector ? Δ : fill(Δ, length(cuts.cuts))
        for (i, cut) in enumerate(cuts.cuts)
            iszero(Δv[i]) && continue
            grad .+= Δv[i] .* _cut_gradient(cut, x̂, u, Kσ0s; u_dep=u_dep)
        end
    end
    return grad
end
function _cuts_notangent(cuts::BucklingCutsFun)
    return Tangent{typeof(cuts)}(;
        solver=NoTangent(), λ̄=NoTangent(), V=NoTangent(), cuts=NoTangent()
    )
end

function ChainRulesCore.rrule(cuts::BucklingCutsFun, x̂::PseudoDensities)
    solver = cuts.solver
    solver.vars .= x̂.x
    solver()
    u = solver.u
    Kσ0s = _element_kσs(solver, u)
    val = cuts(x̂, u, Kσ0s)
    function cuts_pb(Δ)
        return _cuts_notangent(cuts),
        Tangent{typeof(x̂)}(;
            x=_cuts_rrule_pullback(cuts, x̂, u, Kσ0s, Δ; u_dep=true)
        )
    end
    return val, cuts_pb
end
function ChainRulesCore.rrule(cuts::BucklingCutsFun, x::AbstractVector)
    x̂ = PseudoDensities(x)
    solver = cuts.solver
    solver.vars .= x̂.x
    solver()
    u = solver.u
    Kσ0s = _element_kσs(solver, u)
    val = cuts(x̂, u, Kσ0s)
    function cuts_pb(Δ)
        return _cuts_notangent(cuts),
        _cuts_rrule_pullback(cuts, x̂, u, Kσ0s, Δ; u_dep=true)
    end
    return val, cuts_pb
end
function ChainRulesCore.rrule(
    cuts::BucklingCutsFun, x̂::PseudoDensities, u::AbstractVector
)
    Kσ0s = _element_kσs(cuts.solver, u)
    val = cuts(x̂, u, Kσ0s)
    function cuts_pb(Δ)
        return _cuts_notangent(cuts),
        Tangent{typeof(x̂)}(;
            x=_cuts_rrule_pullback(cuts, x̂, u, Kσ0s, Δ; u_dep=false)
        ),
        NoTangent()
    end
    return val, cuts_pb
end
function ChainRulesCore.rrule(
    cuts::BucklingCutsFun, x̂::PseudoDensities, u::AbstractVector,
    Kσ0s::AbstractVector,
)
    val = cuts(x̂, u, Kσ0s)
    function cuts_pb(Δ)
        return _cuts_notangent(cuts),
        Tangent{typeof(x̂)}(;
            x=_cuts_rrule_pullback(cuts, x̂, u, Kσ0s, Δ; u_dep=false)
        ),
        NoTangent(),
        NoTangent()
    end
    return val, cuts_pb
end
