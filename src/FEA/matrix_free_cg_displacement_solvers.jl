abstract type AbstractMatrixFreeSolver <: AbstractDisplacementSolver end

@params mutable struct StaticMatrixFreeDisplacementSolver{T, dim, TP <: AbstractPenalty{T}} <: AbstractDisplacementSolver
    elementinfo::ElementFEAInfo{dim}
    problem::StiffnessTopOptProblem{dim}
    f::AbstractVector{T}
    meandiag::T
    u::AbstractVector{T}
    lhs::AbstractVector{T}
    rhs::AbstractVector{T}
    vars::AbstractVector{T}
    xes
    fixed_dofs
    free_dofs
    penalty::TP
    prev_penalty::TP
    xmin::T
    cg_max_iter::Integer
    tol::T
    cg_statevars::CGStateVariables{T}
    preconditioner
    preconditioner_initialized::Base.RefValue{Bool}
    conv
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, x::StaticMatrixFreeDisplacementSolver) = println("TopOpt matrix free conjugate gradient iterative solver")
StaticMatrixFreeDisplacementSolver(sp, args...; kwargs...) = StaticMatrixFreeDisplacementSolver(whichdevice(sp), sp, args...; kwargs...)

function StaticMatrixFreeDisplacementSolver(
    ::CPU,
    sp::StiffnessTopOptProblem{dim, T};
    conv = DefaultCriteria(), 
    xmin = one(T) / 1000, 
    cg_max_iter = 700, 
    tol = xmin, 
    penalty = PowerPenalty{T}(1), 
    prev_penalty = copy(penalty),
    preconditioner = identity, 
    quad_order = 2,
) where {dim, T}
    prev_penalty = setpenalty(prev_penalty, T(NaN))
    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    if eltype(elementinfo.Kes) <: Symmetric
        f = x -> sumdiag(rawmatrix(x).data)
    else
        f = x -> sumdiag(rawmatrix(x))
    end
    meandiag = mapreduce(f, +, elementinfo.Kes, init = zero(T))
    xes = deepcopy(elementinfo.fes)

    u = zeros(T, ndofs(sp.ch.dh))
    f = similar(u)
    lhs = similar(u)
    rhs = similar(u)
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    cg_statevars = CGStateVariables{eltype(u),typeof(u)}(copy(u), similar(u), similar(u))

    fixed_dofs = sp.ch.prescribed_dofs
    free_dofs = setdiff(1:length(u), fixed_dofs)
    return StaticMatrixFreeDisplacementSolver(elementinfo, sp, f, meandiag, u, lhs, rhs, vars, xes, fixed_dofs, free_dofs, penalty, prev_penalty, xmin, cg_max_iter, tol, cg_statevars, preconditioner, Ref(false), conv)
end

MatrixFreeOperator(solver::StaticMatrixFreeDisplacementSolver) = buildoperator(solver)
function buildoperator(solver::StaticMatrixFreeDisplacementSolver)
    penalty = getpenalty(solver)
    @unpack elementinfo, meandiag, vars, xmin, fixed_dofs, free_dofs, xes, conv = solver
    MatrixFreeOperator(solver.f, elementinfo, meandiag, vars, xes, fixed_dofs, free_dofs, xmin, penalty, conv)
end

function (s::StaticMatrixFreeDisplacementSolver)(
    ; assemble_f = true,
    rhs = assemble_f ? s.f : s.rhs,
    lhs = assemble_f ? s.u : s.lhs,
    kwargs...,
)
    if assemble_f
        assemble_f!(s.f, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin)
    end
    matrix_free_apply2f!(rhs, s.elementinfo, s.meandiag, s.vars, s.problem, getpenalty(s), s.xmin)

    @unpack cg_max_iter, cg_statevars = s
    @unpack preconditioner_initialized, preconditioner, tol = s
    operator = buildoperator(s)

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, operator)
            preconditioner_initialized[] = true
        end
    end
    if preconditioner === identity
        return cg!(lhs, operator, rhs, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false)
    else
        return cg!(lhs, operator, rhs, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false, Pl=preconditioner)
    end
end
