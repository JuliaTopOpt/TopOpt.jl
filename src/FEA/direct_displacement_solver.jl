abstract type AbstractFEASolver end

abstract type AbstractDisplacementSolver <: AbstractFEASolver end
GPUUtils.whichdevice(s::AbstractDisplacementSolver) = whichdevice(s.u)

using Arpack

@params mutable struct DirectDisplacementSolver{T, dim, TP<:AbstractPenalty{T}} <: AbstractDisplacementSolver
    problem::StiffnessTopOptProblem{dim, T}
    globalinfo::GlobalFEAInfo{T}
    elementinfo::ElementFEAInfo{dim, T}
    u::AbstractVector{T}
    lhs::AbstractVector{T}
    rhs::AbstractVector{T}
    vars::AbstractVector{T}
    penalty::TP
    prev_penalty::TP
    xmin::T
end
function DirectDisplacementSolver(sp::StiffnessTopOptProblem{dim, T};
    xmin=T(1)/1000, 
    penalty=PowerPenalty{T}(1), 
    prev_penalty=copy(penalty),
    quad_order=default_quad_order(sp)) where {dim, T}

    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    lhs = similar(u)
    rhs = similar(u)
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind

    prev_penalty = @set prev_penalty.p = T(NaN)
    return DirectDisplacementSolver(sp, globalinfo, elementinfo, u, lhs, rhs, vars, penalty, prev_penalty, xmin)
end
function (s::DirectDisplacementSolver{T})(::Type{Val{safe}} = Val{false}, ::Type{newT} = T; assemble_f = true) where {T, safe, newT}
    rhs = assemble_f ? s.globalinfo.f : s.rhs
    lhs = assemble_f ? s.u : s.lhs
    globalinfo = GlobalFEAInfo(s.globalinfo.K, rhs)
    assemble!(globalinfo, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin, assemble_f = assemble_f)
    K = globalinfo.K
    if safe
        m = meandiag(K)
        for i in 1:size(K,1)
            if K[i,i] ≈ zero(T)
                K[i,i] = m
            end
        end
    end
    try 
        if T === newT
            lhs .= cholesky(Symmetric(K)) \ rhs
        else
            lhs .= cholesky(Symmetric(newT.(K))) \ newT.(rhs)
        end
    catch err
        lhs .= T(NaN)
    end

    nothing
end
