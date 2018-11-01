abstract type AbstractMatrixFreeSolver <: AbstractDisplacementSolver end

mutable struct StaticMatrixFreeDisplacementSolver{T, dim, TS<:StiffnessTopOptProblem{dim, T}, TK<:AbstractMatrix{T}, Tf<:AbstractVector{T}, TKes<:AbstractVector{TK}, Tfes<:AbstractVector{Tf}, Tcload<:AbstractVector{T}, TP<:AbstractPenalty{T}, TPrecond, TOp, TI<:Integer, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}} <: AbstractDisplacementSolver
    problem::TS
    f::Vector{T}
    elementinfo::ElementFEAInfo{dim, T, TK, Tf, TKes, Tfes, Tcload, refshape, TCV, dimless1, TFV}
    meandiag::T
    u::Vector{T}
    vars::Vector{T}
    penalty::TP
    prev_penalty::TP
    xmin::T
    cg_max_iter::TI
    tol::T
    operator::TOp
    cg_statevars::CGStateVariables{T, Vector{T}}
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
            newKes = Symmetric{T, MMatrix{m, m, T, m^2}}[]
            resize!(newKes, length(rawelementinfo.Kes))
            for i in 1:length(rawelementinfo.Kes)
                newKes[i] = Symmetric(MMatrix(rawelementinfo.Kes[i].data))
            end
        else
            newKes = MMatrix{m, m, T, m^2}[]
            resize!(newKes, length(rawelementinfo.Kes))
            for i in 1:length(rawelementinfo.Kes)
                newKes[i] = MMatrix(rawelementinfo.Kes[i])
            end
        end
    else
        newKes = deepcopy(rawelementinfo.Kes)
    end
    elementinfo = ElementFEAInfo(newKes, deepcopy(rawelementinfo.fes), rawelementinfo.fixedload, rawelementinfo.cellvolumes, rawelementinfo.cellvalues, rawelementinfo.facevalues, rawelementinfo.metadata) # cload and cellvalues are shared since they are not overwritten
    meandiag = matrix_free_apply2Kes!(elementinfo, rawelementinfo, sp)

    u = zeros(T, ndofs(sp.ch.dh))
    f = similar(u)
    vars = fill(T(NaN), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    operator = MatrixFreeOperator(elementinfo, meandiag, sp, vars, xmin, penalty)
    cg_statevars = CGStateVariables{eltype(u),typeof(u)}(zeros(u), similar(u), similar(u))

    return StaticMatrixFreeDisplacementSolver(sp, f, rawelementinfo, meandiag, u, vars, penalty, prev_penalty, xmin, cg_max_iter, tol, operator, cg_statevars, preconditioner, Ref(false))
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
