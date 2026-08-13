"""
    SolverResult

Abstract type for the result returned by an FEA solver.
"""
abstract type SolverResult end

# ============================================================================
# New Two-Layered Dispatch System
# ============================================================================

# Physics types - dispatch to different element matrix/assembly functions
"""
    AbstractPhysics

Abstract type for the physics model dispatched on by `GenericFEASolver`.
Subtypes: `LinearElasticity` (structural mechanics, `dim` DOFs/node) and
`HeatTransfer` (heat conduction, 1 DOF/node).
"""
abstract type AbstractPhysics end
"""
    LinearElasticity

Physics tag for structural-mechanics problems. Selects linear-elasticity
element matrices and assembly (dim DOFs per node).
"""
struct LinearElasticity <: AbstractPhysics end      # Structural mechanics (dim DOFs/node)
"""
    HeatTransfer

Physics tag for heat-conduction problems. Selects heat-transfer element
matrices and assembly (1 DOF per node).
"""
struct HeatTransfer <: AbstractPhysics end          # Heat conduction (1 DOF/node)

# Linear solver algorithm types
"""
    AbstractLinearSolver

Abstract type for the linear-system algorithm used inside `GenericFEASolver`.
Subtypes: `DirectSolver` (factorization), `CGAssemblySolver` (CG with an
assembled sparse matrix), `CGMatrixFreeSolver` (matrix-free CG).
"""
abstract type AbstractLinearSolver end
"""
    DirectSolver

Direct linear solver using Cholesky (or QR) factorization. The most robust
option for small to medium problems.
"""
struct DirectSolver <: AbstractLinearSolver end           # Factorization-based (Cholesky/QR)
"""
    CGAssemblySolver

Conjugate-gradient solver with an assembled sparse matrix. Suitable for larger
problems where a factorization is too expensive in memory.
"""
struct CGAssemblySolver <: AbstractLinearSolver end       # CG with assembled matrix
"""
    CGMatrixFreeSolver

Matrix-free conjugate-gradient solver. Avoids assembling the global stiffness
matrix entirely, applying element matrices on the fly. Does not yet support
inhomogeneous Dirichlet BCs.
"""
struct CGMatrixFreeSolver <: AbstractLinearSolver end     # Matrix-free CG

# Export new abstractions
export AbstractPhysics, LinearElasticity, HeatTransfer
export AbstractLinearSolver, DirectSolver, CGAssemblySolver, CGMatrixFreeSolver

# ============================================================================
# Unified GenericFEASolver with Two-Layered Dispatch
# ============================================================================

# CGStateVariables type alias for cleaner code
const CGSV{T,V} = CGStateVariables{T,V}

# Unified solver type with orthogonal physics and linear solver parameters
"""
    GenericFEASolver

Unified FEA solver with orthogonal physics and linear-solver dispatch. Use the
`FEASolver` factory constructor instead of constructing this directly.
"""
mutable struct GenericFEASolver{
    T,
    Physics<:AbstractPhysics,
    Solver<:AbstractLinearSolver,
    TP1<:AbstractPenalty{T},
    TP2<:AbstractTopOptProblem,
    TG<:GlobalFEAInfo{T},
    TE<:ElementFEAInfo,
    Tu<:AbstractVector{T},
    Tc1<:Integer,
    Tc2<:CGSV{T,Tu},
    Tp2,
    Tc3,
} <: AbstractFEASolver
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu           # solution vector
    lhs::Tu
    rhs::Tu
    vars::Tu        # design variables
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    qr::Bool        # use QR instead of Cholesky for Direct solver
    # CG-specific fields
    cg_max_iter::Tc1
    abstol::T
    cg_statevars::Tc2
    preconditioner::Tp2
    preconditioner_initialized::Base.RefValue{Bool}
    conv::Tc3
    # Matrix-free specific fields
    meandiag::T
    fixed_dofs::Vector{Int}
    free_dofs::Vector{Int}
    xes::Vector{Vector{T}}
end

export GenericFEASolver

# Linear solver algorithm dispatch
# These functions dispatch on the linear solver type to solve the system

# Direct solver (factorization-based)
function solve_system!(
    ::Type{DirectSolver},
    solver::GenericFEASolver{T,Physics,DirectSolver},
    K,
    f,
    lhs;
    reuse_fact=false,
    safe=false,
) where {T,Physics}
    if safe
        m = meandiag(K)
        for i in axes(K, 1)
            if K[i, i] ≈ zero(T)
                K[i, i] = m
            end
        end
    end
    if !reuse_fact
        if solver.qr
            solver.globalinfo.qrK = qr(K.data)
        else
            cholK = cholesky(Symmetric(K); check=false)
            if issuccess(cholK)
                solver.globalinfo.cholK = cholK
            else
                @warn "The global matrix is not positive definite. Please check your boundary conditions."
                lhs .= T(NaN)
                return true
            end
        end
    end
    fact = solver.qr ? solver.globalinfo.qrK : solver.globalinfo.cholK
    lhs .= fact \ f
    return false
end

# CG with assembled matrix
function solve_system!(
    ::Type{CGAssemblySolver},
    solver::GenericFEASolver{T,Physics,CGAssemblySolver},
    K,
    f,
    lhs;
    safe=false,
    initially_zero=true,
    kwargs...,
) where {T,Physics}
    if safe
        m = meandiag(K)
        for i in axes(K, 1)
            if K[i, i] ≈ zero(T)
                K[i, i] = m
            end
        end
    end

    @unpack cg_max_iter, abstol, cg_statevars = solver
    @unpack preconditioner, preconditioner_initialized = solver

    _K = K isa Symmetric ? K.data : K
    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, _K)
            preconditioner_initialized[] = true
        end
    end
    # `initially_zero=true` tells IterativeSolvers' cg! that `iszero(lhs)` so
    # it can skip one matvec when forming the initial residual. The caller
    # reuses `solver.u` as `lhs`, which holds the previous solution, so the
    # assumption is violated and CG silently accumulates the new solution on
    # top of the old one. Zero `lhs` to honor the contract.
    if initially_zero
        fill!(lhs, zero(T))
    end
    op = MatrixOperator(_K, f, solver.conv)
    if preconditioner === identity
        return cg!(
            lhs,
            op,
            f;
            abstol=abstol,
            maxiter=cg_max_iter,
            log=false,
            statevars=cg_statevars,
            initially_zero=initially_zero,
        )
    else
        return cg!(
            lhs,
            op,
            f;
            abstol=abstol,
            maxiter=cg_max_iter,
            log=false,
            statevars=cg_statevars,
            initially_zero=initially_zero,
            Pl=preconditioner,
        )
    end
end

# Matrix-free CG
function solve_system!(
    ::Type{CGMatrixFreeSolver},
    solver::GenericFEASolver{T,Physics,CGMatrixFreeSolver},
    K,
    f,
    lhs;
    initially_zero=true,
    kwargs...,
) where {T,Physics}
    @unpack cg_max_iter, abstol, cg_statevars = solver
    @unpack preconditioner, preconditioner_initialized = solver
    @unpack elementinfo, meandiag, vars, xmin, fixed_dofs, free_dofs, xes = solver

    _K = K isa Symmetric ? K.data : K

    # Build matrix-free operator
    penalty = getpenalty(solver)
    operator = MatrixFreeOperator(
        f,
        elementinfo,
        meandiag,
        vars,
        xes,
        fixed_dofs,
        free_dofs,
        xmin,
        penalty,
        solver.conv,
    )

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, _K)
            preconditioner_initialized[] = true
        end
    end
    # See CGAssemblySolver: honor the `initially_zero=true` contract by zeroing
    # `lhs` (which is `solver.u` reused across solves) before cg! accumulates
    # into it.
    if initially_zero
        fill!(lhs, zero(T))
    end
    if preconditioner === identity
        return cg!(
            lhs,
            operator,
            f;
            abstol,
            maxiter=cg_max_iter,
            log=false,
            statevars=cg_statevars,
            initially_zero=initially_zero,
        )
    else
        return cg!(
            lhs,
            operator,
            f;
            abstol,
            maxiter=cg_max_iter,
            log=false,
            statevars=cg_statevars,
            initially_zero=initially_zero,
            Pl=preconditioner,
        )
    end
end

# Unified solver call operator
function (s::GenericFEASolver{T,Physics,Solver})(
    reuse_fact::Bool=false,
    ::Type{Val{safe}}=Val{false};
    assemble_f=true,
    rhs=assemble_f ? s.globalinfo.f : s.rhs,
    lhs=assemble_f ? s.u : s.lhs,
    kwargs...,
) where {T,Physics,Solver,safe}
    # Handle matrix RHS by solving for each column
    if ndims(rhs) == 2 && size(rhs, 2) > 1
        # Multiple RHS columns - solve each one
        # Assemble stiffness matrix (and force vector if assemble_f=true)
        assemble!(
            s.globalinfo,
            s.problem,
            s.elementinfo,
            s.vars,
            getpenalty(s),
            s.xmin;
            assemble_f=assemble_f,
        )

        # Solve for each column of the matrix RHS
        for j in axes(rhs, 2)
            # Get the RHS for this column - use the provided matrix columns
            # Apply boundary conditions to the column
            rhs_j = copy(rhs[:, j])
            apply_zero!(rhs_j, s.problem.ch)

            # Get the view of lhs for this column using @view macro
            lhs_j = @view lhs[:, j]

            # Pass initially_zero only for CG solvers
            if Solver === DirectSolver
                # Filter out kwargs that DirectSolver doesn't accept
                filtered_kwargs = filter(
                    p -> p.first ∉ (:initially_zero, :solver), collect(kwargs)
                )
                solve_system!(
                    Solver,
                    s,
                    s.globalinfo.K,
                    rhs_j,
                    lhs_j;
                    reuse_fact=(j > 1 || reuse_fact),
                    safe=safe,
                    filtered_kwargs...,
                )
            else
                # For CG solvers, start from zero initial guess for each column
                lhs_j .= zero(T)
                solve_system!(
                    Solver,
                    s,
                    s.globalinfo.K,
                    rhs_j,
                    lhs_j;
                    reuse_fact=(j > 1 || reuse_fact),
                    safe=safe,
                    initially_zero=true,
                    kwargs...,
                )
            end
        end
        return nothing
    end

    # Single RHS case (original behavior)
    assemble!(
        s.globalinfo,
        s.problem,
        s.elementinfo,
        s.vars,
        getpenalty(s),
        s.xmin;
        assemble_f=assemble_f,
    )

    # Apply boundary conditions to rhs if needed (only for vectors)
    if !assemble_f && rhs !== s.globalinfo.f && ndims(rhs) == 1
        rhs = copy(rhs)
        apply_zero!(rhs, s.problem.ch)
    end

    # Solve system (physics-independent, solver-algorithm dependent)
    solve_system!(
        Solver, s, s.globalinfo.K, rhs, lhs; reuse_fact=reuse_fact, safe=safe, kwargs...
    )
    return nothing
end

# Show methods
function Base.show(
    io::IO,
    ::MIME{Symbol("text/plain")},
    ::GenericFEASolver{T,LinearElasticity,DirectSolver},
) where {T}
    return println(io, "TopOpt direct structural solver (GenericFEASolver)")
end
function Base.show(
    io::IO, ::MIME{Symbol("text/plain")}, ::GenericFEASolver{T,HeatTransfer,DirectSolver}
) where {T}
    return println(io, "TopOpt direct heat transfer solver (GenericFEASolver)")
end
function Base.show(
    io::IO,
    ::MIME{Symbol("text/plain")},
    ::GenericFEASolver{T,LinearElasticity,CGAssemblySolver},
) where {T}
    return println(io, "TopOpt CG with assembly structural solver (GenericFEASolver)")
end
function Base.show(
    io::IO,
    ::MIME{Symbol("text/plain")},
    ::GenericFEASolver{T,HeatTransfer,CGAssemblySolver},
) where {T}
    return println(io, "TopOpt CG with assembly heat transfer solver (GenericFEASolver)")
end
function Base.show(
    io::IO,
    ::MIME{Symbol("text/plain")},
    ::GenericFEASolver{T,LinearElasticity,CGMatrixFreeSolver},
) where {T}
    return println(io, "TopOpt matrix-free CG structural solver (GenericFEASolver)")
end
function Base.show(
    io::IO,
    ::MIME{Symbol("text/plain")},
    ::GenericFEASolver{T,HeatTransfer,CGMatrixFreeSolver},
) where {T}
    return println(io, "TopOpt matrix-free CG heat transfer solver (GenericFEASolver)")
end

Utilities.getpenalty(solver::AbstractFEASolver) = solver.penalty
function Utilities.setpenalty!(solver::AbstractFEASolver, p)
    solver.prev_penalty = deepcopy(solver.penalty)
    if p isa AbstractPenalty
        solver.penalty = p
    elseif p isa Number
        setpenalty!(solver.penalty, p)
    else
        throw(ArgumentError("Unsupported penalty value $p."))
    end
    return solver
end
Utilities.getprevpenalty(solver::AbstractFEASolver) = solver.prev_penalty

function default_quad_order(problem)
    if TopOptProblems.getdim(problem) == 2 &&
       TopOptProblems.nnodespercell(problem) in (3, 6) ||
        TopOptProblems.getdim(problem) == 3 &&
       TopOptProblems.nnodespercell(problem) in (4, 10)
        return 3
    end
    if TopOptProblems.getgeomorder(problem) == 2
        return 6
    else
        return 4
    end
end

# ============================================================================
# Physics Type Inference from Problem Type
# ============================================================================

# Trait function to infer physics type from problem type
physics_type(::StiffnessTopOptProblem) = LinearElasticity
physics_type(::HeatTransferTopOptProblem) = HeatTransfer

# ============================================================================
# Unified FEASolver Factory with Two-Layered Dispatch
# ============================================================================

# New unified constructor with physics and solver type parameters
"""
    FEASolver(Physics, Solver, problem; kwargs...) -> GenericFEASolver
    FEASolver(Solver, problem; kwargs...) -> GenericFEASolver

Factory constructor for the unified FEA solver. The second form infers the
physics type from the problem type. Keyword arguments: `quad_order`, `xmin`,
`penalty`, `prev_penalty`, `qr`, `cg_max_iter`, `abstol`, `preconditioner`,
`conv`.
"""
function FEASolver(
    ::Type{Physics},
    ::Type{Solver},
    problem::AbstractTopOptProblem;
    quad_order=default_quad_order(problem),
    xmin=nothing,
    penalty=nothing,
    prev_penalty=nothing,
    qr=false,
    # CG options
    cg_max_iter=700,
    abstol=nothing,
    preconditioner=identity,
    # Matrix-free options
    conv=DefaultCriteria(),
    kwargs...,
) where {Physics<:AbstractPhysics,Solver<:AbstractLinearSolver}
    T = TopOptProblems.floattype(problem)
    _xmin = xmin === nothing ? T(1) / 1000 : T(xmin)
    _penalty = penalty === nothing ? PowerPenaltyFun{T}(1) : penalty
    _prev_penalty = prev_penalty === nothing ? deepcopy(_penalty) : prev_penalty
    _abstol = abstol === nothing ? T(1e-7) : T(abstol)

    # Build element matrices based on physics type
    elementinfo = ElementFEAInfo(problem, quad_order, Val{:Static})

    globalinfo = GlobalFEAInfo(problem)

    u = zeros(T, ndofs(problem.ch.dh))
    lhs = similar(u)
    rhs = similar(u)
    # vars stores the full density vector (length = number of elements)
    # Use FixedElementProjectorFun to map free variables to this full vector
    vars = fill(one(T), getncells(problem.ch.dh.grid))

    # Build CG state variables for CG-based solvers
    cg_statevars = CGStateVariables{T,typeof(u)}(copy(u), similar(u), similar(u))

    # Compute meandiag and xes for matrix-free solvers
    if Solver === CGMatrixFreeSolver
        # The matrix-free operator approximates the prescribed-DOF diagonal
        # with a `meandiag` summed over element contributions, which does not
        # match the `meandiag` Ferrite's `apply!` uses to seed the RHS on
        # prescribed DOFs. For homogeneous Dirichlet BCs the RHS is zero there
        # so the mismatch is invisible; for inhomogeneous BCs the prescribed
        # DOFs come out wrong. Fail fast rather than silently returning a
        # corrupted solution.
        if any(!=(0), problem.ch.inhomogeneities)
            throw(
                ArgumentError(
                    "CGMatrixFreeSolver does not yet support inhomogeneous Dirichlet " *
                    "BCs (nonzero prescribed values): the matrix-free meandiag does " *
                    "not match Ferrite's apply! meandiag, so prescribed DOFs are " *
                    "solved to wrong values. Use DirectSolver or CGAssemblySolver " *
                    "for problems with nonzero prescribed temperatures/displacements.",
                ),
            )
        end
        f = x -> sumdiag(rawmatrix(x).data)
        meandiag = mapreduce(f, +, elementinfo.Kes; init=zero(T))
        xes = deepcopy(elementinfo.fes)
        fixed_dofs = problem.ch.prescribed_dofs
        free_dofs = setdiff(eachindex(u), fixed_dofs)
    else
        meandiag = zero(T)
        xes = Vector{Vector{T}}[]
        fixed_dofs = Int[]
        free_dofs = Int[]
    end

    return GenericFEASolver{
        T,
        Physics,
        Solver,
        typeof(_penalty),
        typeof(problem),
        typeof(globalinfo),
        typeof(elementinfo),
        typeof(u),
        typeof(cg_max_iter),
        typeof(cg_statevars),
        typeof(preconditioner),
        typeof(conv),
    }(
        problem,
        globalinfo,
        elementinfo,
        u,
        lhs,
        rhs,
        vars,
        _penalty,
        _prev_penalty,
        _xmin,
        qr,
        cg_max_iter,
        _abstol,
        cg_statevars,
        preconditioner,
        Ref(false),
        conv,
        meandiag,
        fixed_dofs,
        free_dofs,
        xes,
    )
end

# ============================================================================
# Physics-Inferred FEASolver Constructors
# ============================================================================
# These constructors automatically infer the physics type from the problem type

# Direct solver with physics inferred from problem type
function FEASolver(::Type{DirectSolver}, problem::AbstractTopOptProblem; kwargs...)
    return FEASolver(physics_type(problem), DirectSolver, problem; kwargs...)
end

# CG MatrixFree solver with physics inferred from problem type
function FEASolver(::Type{CGAssemblySolver}, problem::AbstractTopOptProblem; kwargs...)
    return FEASolver(physics_type(problem), CGAssemblySolver, problem; kwargs...)
end

function FEASolver(::Type{CGMatrixFreeSolver}, problem::AbstractTopOptProblem; kwargs...)
    return FEASolver(physics_type(problem), CGMatrixFreeSolver, problem; kwargs...)
end

# Export new FEASolver methods
export FEASolver

# simulate convenience wrapper
struct LinearElasticityResult{Tc,Tu}
    comp::Tc
    u::Tu
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::LinearElasticityResult)
    return println(io, "TopOpt linear elasticity result")
end

"""
    simulate(solver, x)

Run a forward FEA solve for the design `x` and return the displacement/temperature
field. Convenience wrapper around the solver call operator.
"""
function simulate(
    problem::StiffnessTopOptProblem,
    topology=ones(getncells(TopOptProblems.getdh(problem).grid));
    round=true,
    hard=true,
    xmin=0.001,
    safe=true,
)
    if round
        if hard
            solver = FEASolver(DirectSolver, problem; xmin=0.0)
        else
            solver = FEASolver(DirectSolver, problem; xmin=xmin)
        end
    else
        solver = FEASolver(DirectSolver, problem; xmin=xmin)
    end
    vars = solver.vars
    fill_vars!(vars, topology; round=round)
    solver(false, Val{safe})
    comp = dot(solver.u, solver.globalinfo.f)
    return LinearElasticityResult(comp, copy(solver.u))
end

function fill_vars!(vars::Array, topology; round)
    if round
        vars .= Base.round.(topology)
    else
        copyto!(vars, topology)
    end
    return vars
end

export simulate
