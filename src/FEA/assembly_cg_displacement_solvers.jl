mutable struct PCGDisplacementSolver{T, dim, TS<:StiffnessTopOptProblem{dim, T}, TK1<:AbstractMatrix{T}, Tf1<:AbstractVector{T}, TK2<:AbstractMatrix{T}, Tf2<:AbstractVector{T}, TKes<:AbstractVector{TK2}, Tfes<:AbstractVector{Tf2}, Tcload<:AbstractVector{T}, TP<:AbstractPenalty{T}, TPrecond, TI<:Integer, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}, Tconv} <: AbstractDisplacementSolver
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
    conv::Tconv
end
function PCGDisplacementSolver(sp::StiffnessTopOptProblem{dim, T};
    conv = DefaultCriteria(),
    xmin=T(1)/1000, 
    cg_max_iter=700, 
    tol=xmin, 
    penalty=PowerPenalty{T}(1), 
    prev_penalty=copy(penalty),
    preconditioner=identity, 
    quad_order=default_quad_order(sp)) where {dim, T, Tconv}

    prev_penalty = @set prev_penalty.p = T(NaN)
    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind
    us = similar(u) .= 0
    cg_statevars = CGStateVariables{eltype(u), typeof(u)}(us, similar(u), similar(u))

    return PCGDisplacementSolver(sp, globalinfo, elementinfo, u, vars, penalty, prev_penalty, xmin, cg_max_iter, tol, cg_statevars, preconditioner, Ref(false), conv)
end

function (s::PCGDisplacementSolver{T})(::Type{Val{safe}} = Val{false}) where {T, safe}
    assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, s.penalty, s.xmin)
    Tconv = typeof(s.conv)
    K, f = s.globalinfo.K, s.globalinfo.f
    if safe
        m = meandiag(K)
        for i in 1:size(K,1)
            if K[i,i] ≈ zero(T)
                K[i,i] = m
            end
        end
    end

    @unpack u, cg_max_iter, tol, cg_statevars = s
    @unpack preconditioner, preconditioner_initialized = s

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
    _K = K isa Symmetric ? K.data : K
    op = MatrixOperator(_K, f, s.conv)
    if preconditioner === identity
        return cg!(u, op, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false)
    else
        return cg!(u, op, f, tol=tol, maxiter=cg_max_iter, log=false, statevars=cg_statevars, initially_zero=false, Pl = preconditioner)
    end
end
