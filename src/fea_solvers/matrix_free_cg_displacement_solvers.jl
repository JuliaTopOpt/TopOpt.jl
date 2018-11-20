abstract type AbstractMatrixFreeSolver <: AbstractDisplacementSolver end

mutable struct StaticMatrixFreeDisplacementSolver{T, dim, TEInfo<:ElementFEAInfo{dim, T}, TS<:StiffnessTopOptProblem{dim, T}, Tv<:AbstractVector{T}, TP<:AbstractPenalty{T}, TI<:Integer, TOp, TStateVars<:CGStateVariables{T}, TPrecond} <: AbstractDisplacementSolver
    elementinfo::TEInfo
    problem::TS
    f::Tv
    meandiag::T
    u::Tv
    vars::Tv
    penalty::TP
    prev_penalty::TP
    xmin::T
    cg_max_iter::TI
    tol::T
    operator::TOp
    cg_statevars::TStateVars
    preconditioner::TPrecond
    preconditioner_initialized::Base.RefValue{Bool}
end

function StaticMatrixFreeDisplacementSolver(sp::StiffnessTopOptProblem{dim, T};
        xmin=T(1)/1000, 
        cg_max_iter=700, 
        tol=xmin, 
        penalty=PowerPenalty{T}(1), 
        prev_penalty=copy(penalty),
        preconditioner=Identity, 
        quad_order=2) where {dim, T}
    
    prev_penalty.p = T(NaN)
    rawelementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    if !(T === BigFloat)
        m = size(rawelementinfo.Kes[1], 1)
        if eltype(rawelementinfo.Kes) <: Symmetric
            newKes = Symmetric{T, SMatrix{m, m, T, m^2}}[]
            resize!(newKes, length(rawelementinfo.Kes))
            map!(x->Symmetric(SMatrix(x.data)), newKes, rawelementinfo.Kes)
        else
            newKes = SMatrix{m, m, T, m^2}[]
            resize!(newKes, length(rawelementinfo.Kes))
            map!(SMatrix, newKes, rawelementinfo.Kes)
        end
    else
        newKes = deepcopy(rawelementinfo.Kes)
    end
    # cload and cellvalues are shared since they are not overwritten
    elementinfo = @set rawelementinfo.Kes = newKes
    elementinfo = @set elementinfo.fes = deepcopy(elementinfo.fes)
    meandiag = matrix_free_apply2Kes!(elementinfo, rawelementinfo, sp)

    u = zeros(T, ndofs(sp.ch.dh))
    f = similar(u)
    vars = fill(T(1), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    operator = MatrixFreeOperator(elementinfo, meandiag, sp, vars, xmin, penalty)
    cg_statevars = CGStateVariables{eltype(u),typeof(u)}(copy(u), similar(u), similar(u))

    return StaticMatrixFreeDisplacementSolver(rawelementinfo, sp, f, meandiag, u, vars, 
        penalty, prev_penalty, xmin, cg_max_iter, tol, operator, cg_statevars, 
        preconditioner, Ref(false))
end

function (s::StaticMatrixFreeDisplacementSolver)()
    assemble_f!(s.f, s.problem, s.elementinfo, s.vars, s.penalty, s.xmin)
    matrix_free_apply2f!(s.f, s.elementinfo, s.meandiag, s.vars, s.problem, s.penalty, s.xmin)

    u = s.u
    f = s.f
    operator = s.operator
    cg_max_iter = s.cg_max_iter
    tol = s.tol
    cg_statevars = s.cg_statevars
    preconditioner = s.preconditioner
    preconditioner_initialized = s.preconditioner_initialized

    if !(preconditioner === Identity)
        if !preconditioner_initialized[]
            UpdatePreconditioner!(preconditioner, operator)
            preconditioner_initialized[] = true
        end
    end
    if preconditioner === Identity
        cg!(u, operator, f, tol, cg_max_iter, Val{false}, cg_statevars, Val{false})
    else
        cg!(u, operator, f, tol, cg_max_iter, Val{false}, cg_statevars, Val{false}, preconditioner)
    end

    #for ind in 1:length(s.dbc.values)
    #    d = s.dbc.dofs[ind]
    #    v = s.dbc.values[ind]
    #    s.u[d] = v
    #end

    s.prev_penalty.p = s.penalty.p
    nothing
end
function (s::StaticMatrixFreeDisplacementSolver)(to)
    @timeit to "Solve system of equations" begin
        assemble_f!(s.f, s.problem, s.elementinfo, s.vars, s.penalty, s.xmin)
        matrix_free_apply2f!(s.f, s.elementinfo, s.meandiag, s.vars, s.problem, s.penalty, s.xmin)

        u = s.u
        f = s.f
        operator = s.operator
        cg_max_iter = s.cg_max_iter
        tol = s.tol
        cg_statevars = s.cg_statevars
        preconditioner = s.preconditioner
        preconditioner_initialized = preconditioner_initialized
    
        if !(preconditioner === Identity)
            if !preconditioner_initialized[]
                UpdatePreconditioner!(preconditioner, operator)
                preconditioner_initialized[] = true
            end
        end
        @timeit to "Conjugate gradient" if preconditioner === Identity
            cg!(u, operator, f, tol, cg_max_iter, Val{false}, cg_statevars, Val{false})
        else
            cg!(u, operator, f, tol, cg_max_iter, Val{false}, cg_statevars, Val{false}, preconditioner)
        end
    
        #for ind in 1:length(s.dbc.values)
        #    d = s.dbc.dofs[ind]
        #    v = s.dbc.values[ind]
        #    s.u[d] = v
        #end
    end
    s.prev_penalty.p = s.penalty.p
    nothing
end

for T in (IterativeSolvers.CGStateVariables, ElementFEAInfo, TopOptProblems.Metadata, StaticMatrixFreeDisplacementSolver, MatrixFreeOperator)
    @eval getfieldnames(::Type{<:$T}) = $(Tuple(fieldnames(T)))
end
cufieldnames(::Type{T}) where {T} = getfieldnames(T)
@eval cufieldnames(::Type{<:ElementFEAInfo}) = $(Tuple(setdiff(fieldnames(ElementFEAInfo), [:cellvalues, :facevalues])))
cufieldnames(::Type{<:StaticMatrixFreeDisplacementSolver}) = (:operator, :cg_statevars)
cufieldnames(::Type{<: MatrixFreeOperator}) = (:elementinfo, :vars)
_cu(s::T, f, fn) where {T} = fn ∈ cufieldnames(T) ? (f isa AbstractArray ? CuArray(f) : cu(f)) : f

for T in (IterativeSolvers.CGStateVariables, ElementFEAInfo, TopOptProblems.Metadata, StaticMatrixFreeDisplacementSolver, MatrixFreeOperator)
    @eval begin
        CuArrays.cu(s::$T) = $T(_cu.((s,), getfield.((s,), getfieldnames($T)), getfieldnames($T))...)
    end
end
