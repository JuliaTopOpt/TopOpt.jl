abstract type AbstractFEASolver end

abstract type AbstractDisplacementSolver <: AbstractFEASolver end

@params mutable struct DirectDisplacementSolver{T,dim,TP<:AbstractPenalty{T}} <:
                       AbstractDisplacementSolver
    problem::StiffnessTopOptProblem{dim,T}
    globalinfo::GlobalFEAInfo{T}
    elementinfo::ElementFEAInfo{dim,T}
    u::AbstractVector{T}
    lhs::AbstractVector{T}
    rhs::AbstractVector{T}
    vars::AbstractVector{T}
    penalty::TP
    prev_penalty::TP
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
    reuse_chol=false,
    rhs=assemble_f ? s.globalinfo.f : s.rhs,
    lhs=assemble_f ? s.u : s.lhs,
    kwargs...,
) where {T,safe,newT}
    globalinfo = s.globalinfo
    N = size(globalinfo.K, 1)
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
    if !reuse_chol
        try
            if T === newT
                if s.qr
                    globalinfo.qrK = qr(K.data)
                else
                    globalinfo.cholK = cholesky(Symmetric(K))
                end
            else
                if s.qr
                    globalinfo.qrK = qr((newT.(K)).data)
                else
                    globalinfo.cholK = cholesky(Symmetric(newT.(K)))
                end
            end
        catch err
            lhs .= T(NaN)
            nans = true
        end
    end
    if !nans
        if T === newT
            if s.qr
                lhs .= globalinfo.qrK \ rhs
            else
                lhs .= globalinfo.cholK \ rhs
            end
        else
            if s.qr
                lhs .= globalinfo.qrK \ newT.(rhs)
            else
                lhs .= globalinfo.cholK \ newT.(rhs)
            end
        end
    end
    return nothing
end
