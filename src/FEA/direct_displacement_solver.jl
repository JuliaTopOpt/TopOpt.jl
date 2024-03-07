abstract type AbstractFEASolver end

abstract type AbstractDisplacementSolver <: AbstractFEASolver end

mutable struct DirectDisplacementSolver{
    T,
    dim,
    TP1<:AbstractPenalty{T},
    TP2<:StiffnessTopOptProblem{dim,T},
    TG<:GlobalFEAInfo{T},
    TE<:ElementFEAInfo{dim,T},
    Tu<:AbstractVector{T},
} <: AbstractDisplacementSolver
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu
    lhs::Tu
    rhs::Tu
    vars::Tu
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    qr::Bool
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, x::DirectDisplacementSolver)
    return println("TopOpt direct solver")
end
function DirectDisplacementSolver(
    sp::StiffnessTopOptProblem{dim,T};
    xmin=T(1) / 1000,
    penalty=PowerPenalty{T}(1),
    prev_penalty=deepcopy(penalty),
    quad_order=default_quad_order(sp),
    qr=false,
) where {dim,T}
    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    lhs = similar(u)
    rhs = similar(u)
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind
    return DirectDisplacementSolver(
        sp, globalinfo, elementinfo, u, lhs, rhs, vars, penalty, prev_penalty, xmin, qr
    )
end
function (s::DirectDisplacementSolver{T})(
    ::Type{Val{safe}}=Val{false},
    ::Type{newT}=T;
    assemble_f=true,
    reuse_fact=false,
    rhs=assemble_f ? s.globalinfo.f : s.rhs,
    lhs=assemble_f ? s.u : s.lhs,
    kwargs...,
) where {T,safe,newT}
    globalinfo = s.globalinfo
    assemble!(
        globalinfo,
        s.problem,
        s.elementinfo,
        s.vars,
        getpenalty(s),
        s.xmin;
        assemble_f=assemble_f,
    )
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
        newK = T === newT ? K : newT.(K)
        if s.qr
            globalinfo.qrK = qr(newK.data)
        else
            cholK = cholesky(Symmetric(K), check = false)
            if issuccess(cholK)
                globalinfo.cholK = cholK
            else
                @warn "The global stiffness matrix is not positive definite. Please check your boundary conditions."
                lhs .= T(NaN)
                nans = true
            end
        end
    end
    nans && return nothing
    new_rhs = T === newT ? rhs : newT.(rhs)
    fact = s.qr ? globalinfo.qrK : globalinfo.cholK
    ldiv!(lhs, fact, new_rhs)
    return nothing
end
