abstract type AbstractMatrixFreeSolver <: AbstractDisplacementSolver end

mutable struct StaticMatrixFreeDisplacementSolver{T, dim, TEInfo <: ElementFEAInfo{dim}, TS<:StiffnessTopOptProblem{dim}, Tv<:AbstractVector{T}, Txes, TDofs, TP<:AbstractPenalty{T}, TI<:Integer, TStateVars<:CGStateVariables{T}, TPrecond} <: AbstractDisplacementSolver
    elementinfo::TEInfo
    problem::TS
    f::Tv
    meandiag::T
    u::Tv
    vars::Tv
    xes::Txes
    fixed_dofs::TDofs
    free_dofs::TDofs
    penalty::TP
    prev_penalty::TP
    xmin::T
    cg_max_iter::TI
    tol::T
    cg_statevars::TStateVars
    preconditioner::TPrecond
    preconditioner_initialized::Base.RefValue{Bool}
end

StaticMatrixFreeDisplacementSolver(sp, args...; kwargs...) = StaticMatrixFreeDisplacementSolver(whichdevice(sp), sp, args...; kwargs...)

const StaticMatrices{m,T} = Union{StaticMatrix{m,m,T}, Symmetric{T, <:StaticMatrix{m,m,T}}}
@generated function sumdiag(K::StaticMatrices{m,T}) where {m,T}
    return reduce((ex1,ex2) -> :($ex1 + $ex2), [:(K[$j,$j]) for j in 1:m])
end

function StaticMatrixFreeDisplacementSolver(::CPU, sp::StiffnessTopOptProblem{dim, T};
        xmin=T(1)/1000, 
        cg_max_iter=700, 
        tol=xmin, 
        penalty=PowerPenalty{T}(1), 
        prev_penalty=copy(penalty),
        preconditioner=identity, 
        quad_order=2) where {dim, T}
    
    prev_penalty = @set prev_penalty.p = T(NaN)
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
    vars = fill(T(1), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    cg_statevars = CGStateVariables{eltype(u),typeof(u)}(copy(u), similar(u), similar(u))

    fixed_dofs = sp.ch.prescribed_dofs
    free_dofs = setdiff(1:length(u), fixed_dofs)
    return StaticMatrixFreeDisplacementSolver(elementinfo, sp, f, meandiag, u, vars, 
        xes, fixed_dofs, free_dofs, penalty, prev_penalty, xmin, cg_max_iter, tol, 
        cg_statevars, preconditioner, Ref(false))
end

MatrixFreeOperator(solver::StaticMatrixFreeDisplacementSolver) = buildoperator(solver)
function buildoperator(solver::StaticMatrixFreeDisplacementSolver)
    penalty = getpenalty(solver)
    @unpack elementinfo, meandiag, vars, xmin, fixed_dofs, free_dofs, xes = solver
    MatrixFreeOperator(elementinfo, meandiag, vars, xes, fixed_dofs, free_dofs, xmin, penalty)
end

function (s::StaticMatrixFreeDisplacementSolver)()
    assemble_f!(s.f, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin)
    matrix_free_apply2f!(s.f, s.elementinfo, s.meandiag, s.vars, s.problem, getpenalty(s), s.xmin)
    
    u = s.u
    f = s.f
    operator = buildoperator(s)
    cg_max_iter = s.cg_max_iter
    tol = s.tol
    cg_statevars = s.cg_statevars
    preconditioner = s.preconditioner
    preconditioner_initialized = s.preconditioner_initialized

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, operator)
            preconditioner_initialized[] = true
        end
    end
    if preconditioner === identity
        cg!(u, operator, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false)
    else
        cg!(u, operator, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false, Pl=preconditioner)
    end

    #for ind in 1:length(s.dbc.values)
    #    d = s.dbc.dofs[ind]
    #    v = s.dbc.values[ind]
    #    s.u[d] = v
    #end

    #s.prev_penalty.p = s.penalty.p
    nothing
end

@define_cu(IterativeSolvers.CGStateVariables, :u, :r, :c)
@define_cu(ElementFEAInfo, :Kes, :fes, :fixedload, :cellvolumes, :metadata, :black, :white, :varind, :cells)
@define_cu(TopOptProblems.Metadata, :cell_dofs, :dof_cells, :node_cells, :node_dofs)
@define_cu(StaticMatrixFreeDisplacementSolver, :f, :problem, :vars, :cg_statevars, :elementinfo, :penalty, :prev_penalty, :u, :fixed_dofs, :free_dofs, :xes)
@define_cu(JuAFEM.ConstraintHandler, :values, :prescribed_dofs, :dh)
@define_cu(JuAFEM.DofHandler, :grid)
@define_cu(JuAFEM.Grid, :cells)
for T in (PointLoadCantilever, HalfMBB, LBeam, TieBeam, InpStiffness)
    @eval @define_cu($T, :ch, :black, :white, :varind)
end

for T in (:PowerPenalty, :RationalPenalty)
    fname = Symbol(:GPU, T)
    @eval @inline CuArrays.cu(p::$T) = $fname(p.p)
end
