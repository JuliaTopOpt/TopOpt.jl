mutable struct PCGDisplacementSolver{T, dim, TS<:StiffnessTopOptProblem{dim, T}, TK1<:AbstractMatrix{T}, Tf1<:AbstractVector{T}, TK2<:AbstractMatrix{T}, Tf2<:AbstractVector{T}, TKes<:AbstractVector{TK2}, Tfes<:AbstractVector{Tf2}, Tcload<:AbstractVector{T}, TP<:AbstractPenalty{T}, TPrecond, TI<:Integer, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}} <: AbstractDisplacementSolver
    problem::TS
    globalinfo::GlobalFEAInfo{T, TK1, Tf1}
    elementinfo::ElementFEAInfo{dim, T, TK2, Tf2, TKes, Tfes, Tcload, refshape, TCV, dimless1, TFV}
    u::Vector{T}
    vars::Vector{T}
    penalty::TP
    prev_penalty::TP
    xmin::T
    cg_max_iter::TI
    tol::T
    cg_statevars::CGStateVariables{T, Vector{T}}
    preconditioner::TPrecond
    preconditioner_initialized::Base.RefValue{Bool}
end
function PCGDisplacementSolver(sp::StiffnessTopOptProblem{dim, T};
    xmin=T(1)/1000, 
    cg_max_iter=700, 
    tol=xmin, 
    penalty=PowerPenalty{T}(1), 
    prev_penalty=copy(penalty),
    preconditioner=identity, 
    quad_order=2) where {dim, T}

    prev_penalty = @set prev_penalty.p = T(NaN)
    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    vars = fill(T(NaN), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind
    us = similar(u) .= 0
    cg_statevars = CGStateVariables{eltype(u), typeof(u)}(us, similar(u), similar(u))

    return PCGDisplacementSolver(sp, globalinfo, elementinfo, u, vars, penalty, prev_penalty, xmin, cg_max_iter, tol, cg_statevars, preconditioner, Ref(false))
end

function (s::PCGDisplacementSolver{T})(to) where {T}
    @timeit to "Assemble" assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, s.penalty, s.xmin)

    K, f = s.globalinfo.K, s.globalinfo.f
    u = s.u
    cg_max_iter = s.cg_max_iter
    tol = s.tol
    cg_statevars = s.cg_statevars
    preconditioner = s.preconditioner
    preconditioner_initialized = s.preconditioner_initialized

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            if K isa Symmetric
                @timeit to "Find Preconditioner" UpdatePreconditioner!(preconditioner, K.data)
            else
                @timeit to "Find Preconditioner" UpdatePreconditioner!(preconditioner, K)
            end
            s.preconditioner_initialized[] = true
        end
    end
    if K isa Symmetric
        if preconditioner === identity
            @timeit to "Solve system of equations" cg!(u, K.data, f, tol, cg_max_iter, Val{false}, cg_statevars, false)
        else
            @timeit to "Solve system of equations" cg!(u, K.data, f, tol, cg_max_iter, Val{false}, cg_statevars, false, preconditioner)
        end
    else
        if preconditioner === identity
            @timeit to "Solve system of equations" cg!(u, K, f, tol, cg_max_iter, Val{false}, cg_statevars, false)
        else
            @timeit to "Solve system of equations" cg!(u, K, f, tol, cg_max_iter, Val{false}, cg_statevars, false, preconditioner)
        end
    end
    #s.prev_penalty.p = s.penalty.p

    nothing
end
function (s::PCGDisplacementSolver{T})(::Type{Val{safe}}=Val{false}) where {T, safe}
    assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, s.penalty, s.xmin)
    K, f = s.globalinfo.K, s.globalinfo.f
    if safe
        m = meandiag(K)
        for i in 1:size(K,1)
            if K[i,i] ≈ zero(T)
                K[i,i] = m
            end
        end
    end

    u = s.u
    cg_max_iter = s.cg_max_iter
    tol = s.tol
    cg_statevars = s.cg_statevars
    preconditioner = s.preconditioner
    preconditioner_initialized = s.preconditioner_initialized

    if !(preconditioner === identity)
        if !preconditioner_initialized[]
            if K isa Symmetric
                UpdatePreconditioner!(preconditioner, K.data)
            else
                UpdatePreconditioner!(preconditioner, K)
            end
            preconditioner_initialized[] = true
        end
    end
    if K isa Symmetric
        _K = K.data
    else
        _K = K
    end
    if preconditioner === identity
        cg!(u, _K, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false)
    else
        cg!(u, _K, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false, Pl = preconditioner)
    end
    #s.prev_penalty.p = s.penalty.p
    
    nothing
end
