"""
    ThermalCompliance{T, TS<:AbstractFEASolver, TC<:AbstractVector{T}, TG<:AbstractVector{T}}

Thermal compliance objective function for heat transfer topology optimization.

# Mathematical Formulation

For steady-state heat conduction with Dirichlet temperature `v` on Γ_D and
heat flux `q` on Γ_N:
    -∇·(k(ρ)∇T) = 0    in Ω
    T = v              on Γ_D
    k∇T·n = q          on Γ_N

where `k(ρ) = k_min + ρ^p (k_0 - k_min)` (SIMP interpolation) and the heat
source `q` is independent of `ρ`.

Thermal compliance is `J = Q^T T` where `Q` is the assembled load vector (the
heat flux contribution on the free degrees of freedom) and `T` is the solved
temperature. With homogeneous Dirichlet BCs (`v = 0`), `J = T^T K(ρ) T`; with
inhomogeneous Dirichlet BCs the two differ and only `Q^T T` is the correct
compliance (the `T^T K T` form leaks the prescribed-temperature energy).

Gradient (adjoint method): solve the condensed system `K_cond λ = -Q_cond`
where `Q_cond` is `Q` zeroed on prescribed DOFs, then
    dJ/dx_e = (λ_e^T Ke T_e) · dρ_e/dx_e

`Q` is independent of `ρ`, so the standard adjoint applies; the only caveat is
that the adjoint state `λ` is not `-T` when the Dirichlet values are nonzero
(because the Dirichlet lift `K(ρ) v` enters the forward residual).

# Usage

```julia
heatflux = Dict{String,Float64}("top" => 100.0)  # heat flux on boundary (W/m²)
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux)
solver = FEASolver(DirectSolver, problem; xmin=0.001)
comp = ThermalCompliance(solver)
val = comp(PseudoDensities(ones(length(solver.vars))))
```
"""
mutable struct ThermalCompliance{
    T, TS<:AbstractFEASolver, TC<:AbstractVector{T}, TG<:AbstractVector{T}
} <: AbstractFunction{T}
    solver::TS
    cell_comp::TC
    grad::TG
    # Workspace for the adjoint solve (allocated lazily on first use)
    adjoint_rhs::TC
    adjoint_sol::TC
end

Utilities.getpenalty(tc::ThermalCompliance) = getpenalty(getsolver(tc))
function Utilities.setpenalty!(tc::ThermalCompliance, p)
    return setpenalty!(getsolver(tc), p)
end
Nonconvex.NonconvexCore.getdim(::ThermalCompliance) = 1

getsolver(tc::ThermalCompliance) = tc.solver

function ThermalCompliance(solver::AbstractFEASolver)
    solver.problem isa HeatTransferTopOptProblem ||
        throw(ArgumentError("ThermalCompliance can only be used with HeatTransferTopOptProblem. Got $(typeof(solver.problem))"))
    T = eltype(solver.vars)
    nel = getncells(solver.problem.ch.dh.grid)
    cell_comp = zeros(T, nel)
    grad = copy(cell_comp)
    n = ndofs(solver.problem.ch.dh)
    adjoint_rhs = zeros(T, n)
    adjoint_sol = zeros(T, n)
    return ThermalCompliance(solver, cell_comp, grad, adjoint_rhs, adjoint_sol)
end

function (tc::ThermalCompliance)(x::AbstractVector)
    @warn "A vector input was passed in to the thermal compliance function. It will be assumed to be the filtered, unpenalised and uninterpolated pseudo-densities. Please use the `PseudoDensities` constructor to wrap the input vector to avoid ambiguity."
    return tc(PseudoDensities(x))
end

function (tc::ThermalCompliance{T})(x::PseudoDensities) where {T}
    solver = getsolver(tc)
    solver.vars .= x.x
    solver()
    return compute_thermal_compliance!(tc, solver)
end

function ChainRulesCore.rrule(tc::ThermalCompliance, x::PseudoDensities)
    out = tc(x)
    out_grad = copy(tc.grad)
    return out, Δ -> (nothing, Tangent{typeof(x)}(; x=out_grad * Δ))
end

"""
    compute_thermal_compliance!(tc, solver)

Compute thermal compliance `J = Q^T T` and its adjoint gradient.

`Q` is the non-penalized load vector (`elementinfo.fixedload`), available
before `apply!` modifies the assembled RHS. `T` is the solved temperature
(`solver.u`). Only the free DOFs contribute to `J` because `Q` is zero on
prescribed DOFs (heat flux is a Neumann BC; Dirichlet DOFs carry no entry in
`Q`), so `Q^T T` over all DOFs equals `Q_f^T T_f`.

Gradient: the adjoint solve `K_cond λ = -Q_cond` (`Q_cond` = `Q` zeroed on
prescribed DOFs) gives `λ`, then `grad_e = (λ_e^T Ke T_e) · dρ_e/dx_e`.

For homogeneous Dirichlet BCs, `λ = -T` and this reduces to the familiar
`grad_e = -dρ_e/dx_e · T_e^T Ke T_e`. The general form is required whenever
the prescribed temperatures are nonzero.
"""
function compute_thermal_compliance!(tc, solver)
    @unpack cell_comp, grad, adjoint_rhs, adjoint_sol = tc
    @unpack elementinfo, u, xmin = solver
    Kes = elementinfo.Kes
    cell_dofs = solver.problem.metadata.cell_dofs
    penalty = getpenalty(tc)

    Q = elementinfo.fixedload

    # Forward solve already done; J = Q^T T (free DOFs only, but Q is zero on
    # prescribed DOFs so the full dot product is identical).
    obj = dot(Q, u)

    # Adjoint solve: K_cond λ = -Q_cond, where Q_cond zeros prescribed DOFs.
    # The prescribed-DOF rows of K_cond carry `meandiag` on the diagonal and
    # -Q_cond is zero there, so λ vanishes on prescribed DOFs automatically.
    ch = solver.problem.ch
    @inbounds for i in eachindex(adjoint_rhs)
        adjoint_rhs[i] = -Q[i]
    end
    pdofs = ch.prescribed_dofs
    @inbounds for d in pdofs
        adjoint_rhs[d] = zero(eltype(adjoint_rhs))
    end

    solve_adjoint!(solver, adjoint_sol, adjoint_rhs)

    # Per-element gradient: grad_e = (λ_e^T Ke T_e) · dρ_e/dx_e
    @inbounds for i in 1:size(cell_dofs, 2)
        Ke = rawmatrix(Kes[i])
        cell_energy = zero(obj)
        for w in 1:size(Ke, 2)
            for v in 1:size(Ke, 1)
                cell_energy += adjoint_sol[cell_dofs[v, i]] * Ke[v, w] * u[cell_dofs[w, i]]
            end
        end
        _, dρe = get_ρ_dρ(solver.vars[i], penalty, xmin)
        cell_comp[i] = cell_energy
        grad[i] = dρe * cell_energy
    end

    return obj
end

"""
    solve_adjoint!(solver, lhs, rhs)

Solve the condensed system `K_cond lhs = rhs` using the same linear solver as
the forward pass. `K_cond` is the already-assembled, boundary-condition-applied
matrix stored in `solver.globalinfo` (for `DirectSolver` and
`CGAssemblySolver`); for `CGMatrixFreeSolver` the matrix-free operator is
rebuilt and solved with CG.
"""
function solve_adjoint! end

function solve_adjoint!(
    solver::GenericFEASolver{T,Physics,DirectSolver}, lhs, rhs
) where {T,Physics}
    K = solver.globalinfo.K
    lhs .= K \ rhs
    return nothing
end

function solve_adjoint!(
    solver::GenericFEASolver{T,Physics,CGAssemblySolver}, lhs, rhs
) where {T,Physics}
    K = solver.globalinfo.K
    _K = K isa Symmetric ? K.data : K
    op = MatrixOperator(_K, rhs, solver.conv)
    # Zero `lhs` so the `initially_zero=true` contract with cg! holds (lhs is
    # `adjoint_sol`, reused across ThermalCompliance evaluations).
    fill!(lhs, zero(T))
    if solver.preconditioner === identity
        cg!(lhs, op, rhs; abstol=solver.abstol, maxiter=solver.cg_max_iter,
            log=false, statevars=solver.cg_statevars, initially_zero=true)
    else
        if !solver.preconditioner_initialized[]
            UpdatePreconditioner!(solver.preconditioner, _K)
            solver.preconditioner_initialized[] = true
        end
        cg!(lhs, op, rhs; abstol=solver.abstol, maxiter=solver.cg_max_iter,
            log=false, statevars=solver.cg_statevars, initially_zero=true,
            Pl=solver.preconditioner)
    end
    return nothing
end

function solve_adjoint!(
    solver::GenericFEASolver{T,Physics,CGMatrixFreeSolver}, lhs, rhs
) where {T,Physics}
    @unpack elementinfo, meandiag, vars, xes, fixed_dofs, free_dofs = solver
    penalty = getpenalty(solver)
    operator = MatrixFreeOperator(
        rhs, elementinfo, meandiag, vars, xes,
        fixed_dofs, free_dofs, solver.xmin, penalty, solver.conv
    )
    fill!(lhs, zero(T))
    if solver.preconditioner === identity
        cg!(lhs, operator, rhs; abstol=solver.abstol, maxiter=solver.cg_max_iter,
            log=false, statevars=solver.cg_statevars, initially_zero=true)
    else
        _K = solver.globalinfo.K
        _K = _K isa Symmetric ? _K.data : _K
        if !solver.preconditioner_initialized[]
            UpdatePreconditioner!(solver.preconditioner, _K)
            solver.preconditioner_initialized[] = true
        end
        cg!(lhs, operator, rhs; abstol=solver.abstol, maxiter=solver.cg_max_iter,
            log=false, statevars=solver.cg_statevars, initially_zero=true,
            Pl=solver.preconditioner)
    end
    return nothing
end

export ThermalCompliance