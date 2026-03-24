abstract type SolverResult end

# ============================================================================
# New Two-Layered Dispatch System
# ============================================================================

# Physics types - dispatch to different element matrix/assembly functions
abstract type AbstractPhysics end
struct LinearElasticity <: AbstractPhysics end      # Structural LinearElasticity (dim DOFs/node)
struct HeatTransfer <: AbstractPhysics end  # Heat conduction (1 DOF/node)

# Linear solver algorithm types
abstract type AbstractLinearSolver end
struct DirectSolver <: AbstractLinearSolver end           # Factorization-based (Cholesky/QR)
struct CGAssemblySolver <: AbstractLinearSolver end       # CG with assembled matrix
struct CGMatrixFreeSolver <: AbstractLinearSolver end     # Matrix-free CG

# Export new abstractions
export AbstractPhysics, LinearElasticity, HeatTransfer
export AbstractLinearSolver, DirectSolver, CGAssemblySolver, CGMatrixFreeSolver

# Export shared abstractions
export supports_reuse_fact

# Trait for solver capabilities - factorization-based solvers support reuse
supports_reuse_fact(::Type{<:AbstractFEASolver}) = false

# Common initialization for direct solvers
function init_direct_solver(
    sp::AbstractTopOptProblem,
    quad_order::Int,
    xmin::T,
    penalty::AbstractPenalty{T},
    prev_penalty::AbstractPenalty{T},
    qr::Bool,
) where {T}
    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    lhs = similar(u)
    rhs = similar(u)
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    return elementinfo, globalinfo, u, lhs, rhs, vars
end

# Common factorization and solve logic for direct solvers
function direct_solve_core!(
    solver::AbstractFEASolver,
    globalinfo::GlobalFEAInfo,
    lhs::AbstractVector,
    rhs::AbstractVector,
    reuse_fact::Bool,
    safe::Bool,
    matrix_name::String,
)
    T = eltype(solver)
    K = globalinfo.K
    if safe
        m = meandiag(K)
        for i in 1:size(K, 1)
            if K[i, i] ≈ zero(T)
                K[i, i] = m
            end
        end
    end
    nans = false
    if !reuse_fact
        if solver.qr
            globalinfo.qrK = qr(K.data)
        else
            cholK = cholesky(Symmetric(K); check=false)
            if issuccess(cholK)
                globalinfo.cholK = cholK
            else
                @warn "The global $matrix_name matrix is not positive definite. Please check your boundary conditions."
                lhs .= T(NaN)
                nans = true
            end
        end
    end
    nans && return true
    fact = solver.qr ? globalinfo.qrK : globalinfo.cholK
    lhs .= fact \ rhs
    return false
end

# ============================================================================
# Unified GenericFEASolver with Two-Layered Dispatch
# ============================================================================

# CGStateVariables type alias for cleaner code
const CGSV{T,V} = CGStateVariables{T,V}

# Unified solver type with orthogonal physics and linear solver parameters
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

# Physics-specific matrix building dispatch
# These functions dispatch on physics type to build the correct element matrices
function build_element_matrices(::Type{LinearElasticity}, problem::StiffnessTopOptProblem{dim,T}, quad_order) where {dim,T}
    return make_Kes_and_fes(problem, quad_order, Val{:Static})
end

# Linear solver algorithm dispatch
# These functions dispatch on the linear solver type to solve the system

# Direct solver (factorization-based)
function solve_system!(::Type{DirectSolver}, solver::GenericFEASolver{T,Physics,DirectSolver}, K, f, lhs;
                     reuse_fact=false, safe=false) where {T,Physics}
    if safe
        m = meandiag(K)
        for i in 1:size(K, 1)
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
function solve_system!(::Type{CGAssemblySolver}, solver::GenericFEASolver{T,Physics,CGAssemblySolver}, K, f, lhs;
                      safe=false, kwargs...) where {T,Physics}
    if safe
        m = meandiag(K)
        for i in 1:size(K, 1)
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
            initially_zero=false,
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
            initially_zero=false,
            Pl=preconditioner,
        )
    end
end

# Matrix-free CG
function solve_system!(::Type{CGMatrixFreeSolver}, solver::GenericFEASolver{T,Physics,CGMatrixFreeSolver}, K, f, lhs;
                      kwargs...) where {T,Physics}
    @unpack cg_max_iter, abstol, cg_statevars = solver
    @unpack preconditioner, preconditioner_initialized = solver
    @unpack elementinfo, meandiag, vars, xmin, fixed_dofs, free_dofs, xes = solver

    # Build matrix-free operator
    penalty = getpenalty(solver)
    operator = MatrixFreeOperator(
        f, elementinfo, meandiag, vars, xes,
        fixed_dofs, free_dofs, xmin, penalty, solver.conv
    )

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, operator)
            preconditioner_initialized[] = true
        end
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
            initially_zero=false,
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
            initially_zero=false,
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
    kwargs...
) where {T,Physics,Solver,safe}
    # Handle matrix RHS by solving for each column
    if ndims(rhs) == 2 && size(rhs, 2) > 1
        # Multiple RHS columns - solve each one
        if assemble_f
            # Assemble once
            assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin; assemble_f=true)
        end
        # Solve for each column
        for j in 1:size(rhs, 2)
            @views begin
                rhs_j = Vector(rhs[:, j])  # Convert sparse to dense vector
                lhs_j = lhs[:, j]
                solve_system!(Solver, s, s.globalinfo.K, rhs_j, lhs_j;
                             reuse_fact=(j > 1 || reuse_fact), safe=safe, kwargs...)
            end
        end
        return nothing
    end

    # Single RHS case (original behavior)
    assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin; assemble_f=assemble_f)

    # Apply boundary conditions to rhs if needed (only for vectors)
    if !assemble_f && rhs !== s.globalinfo.f && ndims(rhs) == 1
        rhs = copy(rhs)
        apply_zero!(rhs, s.problem.ch)
    end

    # Solve system (physics-independent, solver-algorithm dependent)
    solve_system!(Solver, s, s.globalinfo.K, rhs, lhs; reuse_fact=reuse_fact, safe=safe, kwargs...)
    return nothing
end

# Show methods
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::GenericFEASolver{T,LinearElasticity,DirectSolver}) where {T}
    return println("TopOpt direct structural solver (GenericFEASolver)")
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::GenericFEASolver{T,HeatTransfer,DirectSolver}) where {T}
    return println("TopOpt direct heat transfer solver (GenericFEASolver)")
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::GenericFEASolver{T,LinearElasticity,CGAssemblySolver}) where {T}
    return println("TopOpt CG with assembly structural solver (GenericFEASolver)")
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::GenericFEASolver{T,LinearElasticity,CGMatrixFreeSolver}) where {T}
    return println("TopOpt matrix-free CG structural solver (GenericFEASolver)")
end

Utilities.getpenalty(solver::AbstractFEASolver) = solver.penalty
function Utilities.setpenalty!(solver::AbstractFEASolver, p)
    solver.prev_penalty = deepcopy(solver.penalty)
    if p isa AbstractPenalty
        solver.penalty = p
    elseif p isa Number
        setpenalty!(solver.penalty, p)
    else
        throw("Unsupported penalty value $p.")
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
    kwargs...
) where {Physics<:AbstractPhysics,Solver<:AbstractLinearSolver}
    T = TopOptProblems.floattype(problem)
    _xmin = xmin === nothing ? T(1)/1000 : T(xmin)
    _penalty = penalty === nothing ? PowerPenalty{T}(1) : penalty
    _prev_penalty = prev_penalty === nothing ? deepcopy(_penalty) : prev_penalty
    _abstol = abstol === nothing ? T(1e-7) : T(abstol)

    # Build element matrices based on physics type
    if Physics === LinearElasticity
        elementinfo = ElementFEAInfo(problem, quad_order, Val{:Static})
    elseif Physics === HeatTransfer
        elementinfo = ElementFEAInfo(problem, quad_order, Val{:Static})
    else
        error("Physics type $Physics not yet implemented")
    end

    globalinfo = GlobalFEAInfo(problem)

    u = zeros(T, ndofs(problem.ch.dh))
    lhs = similar(u)
    rhs = similar(u)
    vars = fill(one(T), getncells(problem.ch.dh.grid) - sum(problem.black) - sum(problem.white))

    # Build CG state variables for CG-based solvers
    cg_statevars = CGStateVariables{T,typeof(u)}(copy(u), similar(u), similar(u))

    # Compute meandiag and xes for matrix-free solvers
    if Solver === CGMatrixFreeSolver
        if eltype(elementinfo.Kes) <: Symmetric
            f = x -> sumdiag(rawmatrix(x).data)
        else
            f = x -> sumdiag(rawmatrix(x))
        end
        meandiag = mapreduce(f, +, elementinfo.Kes; init=zero(T))
        xes = deepcopy(elementinfo.fes)
        fixed_dofs = problem.ch.prescribed_dofs
        free_dofs = setdiff(1:length(u), fixed_dofs)
    else
        meandiag = zero(T)
        xes = Vector{Vector{T}}[]
        fixed_dofs = Int[]
        free_dofs = Int[]
    end

    return GenericFEASolver{T,Physics,Solver,typeof(_penalty),typeof(problem),
                            typeof(globalinfo),typeof(elementinfo),typeof(u),
                            typeof(cg_max_iter),typeof(cg_statevars),typeof(preconditioner),typeof(conv)}(
        problem, globalinfo, elementinfo, u, lhs, rhs, vars,
        _penalty, _prev_penalty, _xmin, qr,
        cg_max_iter, _abstol, cg_statevars, preconditioner, Ref(false), conv,
        meandiag, fixed_dofs, free_dofs, xes
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
