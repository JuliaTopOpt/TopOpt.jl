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

"""
    buckling(solver::AbstractFEASolver)

The solver-state convenience method of [`TopOptProblems.buckling`](@ref):
returns `(K, Kσ)` for the solver's *current* design, using the solver's
own displacement solution `solver.u` as the prebuckling field (the solver
must have been called at the current design). No solve and no
factorization happen here, so the cost profile is identical for direct
and iterative (CG) displacement solvers — the field the design analysis
used is the field the buckling analysis uses.
"""
function buckling(solver::AbstractFEASolver)
    problem = solver.problem
    return buckling(
        problem, solver.globalinfo, solver.elementinfo, solver.vars,
        getpenalty(solver), solver.xmin; u=solver.u,
    )
end

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
    solver.vars .= x.x
    solver()
    K, Kσ = buckling(solver)
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

"""
    BucklingElementCutsFun(solver::AbstractFEASolver, λ̄, V; active=nothing, c̄=nothing)

The element-based relaxation of the buckling constraint bundled with the
global cuts (and optionally the compliance): one evaluation solves the
prebuckling displacement field and assembles the per-cell stress state
*once*, then returns the tuple

    (c(x)/c̄ − 1 if c̄ !== nothing,
     P_e for e ∈ active..., vᵢᵀ(K + λ̄Kσ)vᵢ for vᵢ ∈ V...),

where `P_e = ρ_e·(K0e + λ̄·G0e)` is element `e`'s contribution to the
global buckling matrix `K + λ̄Kσ` (`K0e` the element stiffness, `G0e` the
unpenalized geometric stiffness from the shared stress state). Enforcing
`P_e ⪰ 0` for every element is the local (element-wise) relaxation of
`K + λ̄Kσ ⪰ 0` — sufficient, conservative, and blind to global mode
shapes; the cuts capture the remaining global violations. Elements with
`ρ_e ≈ 0` satisfy it automatically, and tension elements (`G0e ⪰ 0`) do
too, so in practice only compression-carrying solid material is charged.

Declare the bundle as ONE function-in-set constraint:

```julia
add_set_constraint!(
    model, x -> bec(filter(x)),
    SetProduct(NonconvexCore.AbstractSet[
        LessThan(0.0),                                # when c̄ !== nothing
        fill(PSDCone(8), length(active))...,
        Nonnegatives(length(V)),
    ]),
)
```

which NonconvexAugLagLab (with NonconvexSemidefinite loaded) evaluates as
one group: the projected augmented Lagrangian charges the PSD components
and the classical ALM the cut components, all sharing the one solve. The
reverse rule assembles the gradients of every component from a *single*
adjoint solve — the per-cell cotangents of the stress matrices (the
cuts' `λ̄·ρ_e·Δcᵢ·vᵢvᵢᵀ` and the elements' `λ̄·ρ_e·ΔP_e`) are summed into
one load.

`active` selects the elements contributing PSD components; defaults to
every element with a free design variable (passive solid/void excluded).
`V` may be empty (the local relaxation alone). With `c̄` given, the first
output component is the normalized compliance `c(x)/c̄ − 1` (the
displacement solve it shares with everything else), to be paired with a
`LessThan(0.0)` component. The cut values are the raw quadratic forms
`vᵢᵀ(K + λ̄Kσ)vᵢ` — pair the last component with `Nonnegatives` to enforce
them ≥ 0.
"""
struct BucklingElementCutsFun{TS<:AbstractFEASolver,T,TV,TC,TA,TCB}
    solver::TS
    λ̄::T
    V::TV
    cuts::TC
    active::TA
    c̄::TCB
end
function BucklingElementCutsFun(
    solver::AbstractFEASolver, λ̄, V; active=nothing, c̄=nothing
)
    cuts = [BucklingCutFun(solver, λ̄, v) for v in V]
    problem = solver.problem
    if active === nothing
        m = getncells(problem.ch.dh.grid)
        pas = applicable(getpassive_solid, problem) ?
              union(getpassive_solid(problem), getpassive_void(problem)) :
              Int[]
        active = [e for e in 1:m if !(e in pas)]
    end
    return BucklingElementCutsFun(solver, λ̄, V, cuts, active, c̄)
end

# one bundle evaluation from the shared solve + stress state
function _bec_tuple(bec::BucklingElementCutsFun, x̂::PseudoDensities, u, Kσ0s, ρs)
    dh = bec.solver.problem.ch.dh
    gd = zeros(Int, ndofs_per_cell(dh))
    m = length(Kσ0s)
    K = bec.solver.globalinfo.K
    out = Any[]
    if bec.c̄ !== nothing
        push!(out, dot(u, bec.solver.globalinfo.f) / bec.c̄ - 1)
    end
    for e in bec.active
        K0e = Matrix(rawmatrix(bec.solver.elementinfo.Kes[e]))
        push!(out, ρs[e] .* (K0e + bec.λ̄ .* Kσ0s[e]))
    end
    if !isempty(bec.V)
        # the cuts are ONE vector component (paired with Nonnegatives(n)):
        # n scalar entries would misalign with the set components.
        cutvals = zeros(length(bec.V))
        for (i, v) in enumerate(bec.V)
            val = dot(v, K * v)
            for e in 1:m
                celldofs!(gd, dh, e)
                val += bec.λ̄ * ρs[e] * dot(v[gd], Kσ0s[e], v[gd])
            end
            cutvals[i] = val
        end
        push!(out, cutvals)
    end
    # a plain Vector on purpose: FunctionInSet keeps vector outputs as
    # vectors, so products with thousands of element components never
    # materialize a giant tuple type.
    return out
end

# zero cotangents: ChainRules zeros AND plain scalar zeros (manual
# callers may pass untyped zero vectors).
_cotzero(Δx) = Δx isa ChainRulesCore.AbstractZero ||
               (Δx isa Number && iszero(Δx))

function _bec_notangent(bec::BucklingElementCutsFun)
    return Tangent{typeof(bec)}(;
        solver=NoTangent(), λ̄=NoTangent(), V=NoTangent(),
        cuts=NoTangent(), active=NoTangent(), c̄=NoTangent(),
    )
end

function (bec::BucklingElementCutsFun)(x̂::PseudoDensities)
    solver = bec.solver
    solver.vars .= x̂.x
    solver()
    u = solver.u
    penalty = getpenalty(solver)
    xmin = solver.xmin
    m = getncells(solver.problem.ch.dh.grid)
    Kσ0s = _element_kσs(solver, u)
    ρs = [get_ρ_dρ(x̂.x[e], penalty, xmin)[1] for e in 1:m]
    # compliance (shared u), element submatrices and cuts (shared stress):
    # all components of one evaluation come from the same state.
    return _bec_tuple(bec, x̂, u, Kσ0s, ρs)
end
function (bec::BucklingElementCutsFun)(x::AbstractVector)
    return bec(PseudoDensities(x))
end

function ChainRulesCore.rrule(
    bec::BucklingElementCutsFun, x̂::PseudoDensities
)
    solver = bec.solver
    solver.vars .= x̂.x
    solver()
    u = solver.u
    problem = solver.problem
    penalty = getpenalty(solver)
    xmin = solver.xmin
    dh = problem.ch.dh
    cellvalues = solver.elementinfo.cellvalues
    Kσ0s = _element_kσs(solver, u)
    m = length(Kσ0s)
    gd = zeros(Int, ndofs_per_cell(dh))
    ρs = [get_ρ_dρ(x̂.x[e], penalty, xmin)[1] for e in 1:m]
    K0act = [Matrix(rawmatrix(solver.elementinfo.Kes[e])) for e in bec.active]
    actidx = Dict(e => k for (k, e) in enumerate(bec.active))
    val = _bec_tuple(bec, x̂, u, Kσ0s, ρs)
    nc = bec.c̄ === nothing ? 0 : 1   # compliance component offset
    function bec_pb(Δ)
        Δ = ChainRulesCore.unthunk(Δ)
        grad = zeros(eltype(x̂.x), m)
        if Δ isa ChainRulesCore.AbstractZero
            return _bec_notangent(bec), Tangent{typeof(x̂)}(; x=grad)
        end
        na = length(bec.active)
        ncut = length(bec.V)
        Δcomp = nc == 1 ? ChainRulesCore.unthunk(Δ[1]) :
                ChainRulesCore.NoTangent()
        Δcuts = ncut > 0 ? ChainRulesCore.unthunk(Δ[nc + na + 1]) :
                ChainRulesCore.NoTangent()
        cv = Ferrite.CellCache(dh)
        # ONE combined adjoint load: the per-cell cotangents on G0e from
        # the element components (λ̄ρ_eΔP_e) and the cut components
        # (Δcᵢ·λ̄ρ_e·vᵢvᵢᵀ), all absorbed λ̄ so a single w = K⁻¹load serves
        # every component's u-chain. The compliance component needs no
        # load: its design derivative −uᵀ(∂K/∂x̂)u is self-adjoint and is
        # added directly from the same shared u.
        load = zeros(size(u))
        for e in 1:m
            ΔG = zeros(eltype(u), ndofs_per_cell(dh), ndofs_per_cell(dh))
            k = get(actidx, e, 0)
            if k != 0
                ΔP = ChainRulesCore.unthunk(Δ[nc + k])
                !_cotzero(ΔP) && (ΔG .+= bec.λ̄ .* ρs[e] .* ΔP)
            end
            if !_cotzero(Δcuts)
                for (i, v) in enumerate(bec.V)
                    Δc = ChainRulesCore.unthunk(Δcuts[i])
                    _cotzero(Δc) && continue
                    celldofs!(gd, dh, e)
                    ΔG .+= Δc .* bec.λ̄ .* ρs[e] .* (v[gd] * v[gd]')
                end
            end
            iszero(ΔG) && continue
            celldofs!(gd, dh, e)
            Ferrite.reinit!(cv, e)
            reinit!(cellvalues, cv)
            _, _, _, Δue = ChainRulesCore.rrule(
                element_kσ, problem, cellvalues, u[gd]
            )[2](ΔG)
            load[gd] .+= Δue
        end
        solver.rhs .= load
        solver(; reuse_fact=true, assemble_f=false)
        w = solver.lhs
        for (k, e) in enumerate(bec.active)
            ΔP = ChainRulesCore.unthunk(Δ[nc + k])
            celldofs!(gd, dh, e)
            ue = u[gd]
            we = w[gd]
            K0e = K0act[k]
            G0e = Kσ0s[e]
            t = zero(eltype(x̂.x))
            if !_cotzero(ΔP)
                t += dot(ΔP, K0e + bec.λ̄ .* G0e)
            end
            if !_cotzero(Δcomp)
                # compliance (shared u): d/dx̂[c/c̄] = −uᵀ(∂K/∂x̂)u / c̄
                t -= Δcomp * dot(ue, K0e, ue) / bec.c̄
            end
            if !_cotzero(Δcuts)
                for (i, v) in enumerate(bec.V)
                    Δc = ChainRulesCore.unthunk(Δcuts[i])
                    _cotzero(Δc) && continue
                    ve = v[gd]
                    t += Δc * (dot(ve, K0e, ve) + bec.λ̄ * dot(ve, G0e, ve))
                end
            end
            _, dρ = get_ρ_dρ(x̂.x[e], penalty, xmin)
            grad[e] = dρ * (t - dot(we, K0e, ue))
        end
        return _bec_notangent(bec), Tangent{typeof(x̂)}(; x=grad)
    end
    return val, bec_pb
end
