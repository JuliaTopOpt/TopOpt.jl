abstract type AbstractFEASolver end

abstract type AbstractDisplacementSolver <: AbstractFEASolver end
GPUUtils.whichdevice(s::AbstractDisplacementSolver) = whichdevice(s.u)

mutable struct DirectDisplacementSolver{T, dim, TS<:StiffnessTopOptProblem{dim, T}, TK1<:AbstractMatrix{T}, Tf1<:AbstractVector{T}, TK2<:AbstractMatrix{T}, Tf2<:AbstractVector{T}, TKes<:AbstractVector{TK2}, Tfes<:AbstractVector{Tf2}, Tcload<:AbstractVector{T}, TP<:AbstractPenalty{T}, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}} <: AbstractDisplacementSolver
    problem::TS
    globalinfo::GlobalFEAInfo{T, TK1, Tf1}
    elementinfo::ElementFEAInfo{dim, T, TK2, Tf2, TKes, Tfes, Tcload, refshape, TCV, dimless1, TFV}
    u::Vector{T}
    vars::Vector{T}
    penalty::TP
    prev_penalty::TP
    xmin::T
end
function DirectDisplacementSolver(sp::StiffnessTopOptProblem{dim, T};
    xmin=T(1)/1000, 
    penalty=PowerPenalty{T}(1), 
    prev_penalty=copy(penalty),
    quad_order=2) where {dim, T}

    elementinfo = ElementFEAInfo(sp, quad_order, Val{:Static})
    globalinfo = GlobalFEAInfo(sp)
    u = zeros(T, ndofs(sp.ch.dh))
    vars = fill(T(NaN), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind

    prev_penalty = @set prev_penalty.p = T(NaN)
    return DirectDisplacementSolver(sp, globalinfo, elementinfo, u, vars, penalty, prev_penalty, xmin)
end
function (s::DirectDisplacementSolver{T})(::Type{Val{safe}}=Val{false}, ::Type{newT}=T) where {T, safe, newT}
    assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin)
    K, f = s.globalinfo.K, s.globalinfo.f
    if safe
        m = meandiag(K)
        for i in 1:size(K,1)
            if K[i,i] ≈ zero(T)
                K[i,i] = m
            end
        end
    end
    s.u .=  try 
        if T === newT
            Symmetric(K) \ f
        else
            Symmetric(newT.(K)) \ newT.(f)
        end
    catch
        T(NaN)
    end
 
    nothing
end
function (s::DirectDisplacementSolver{T})(to, ::Type{Val{safe}}=Val{false}, ::Type{newT}=T) where {T, safe, newT}
    @timeit to "Assemble" assemble!(s.globalinfo, s.problem, s.elementinfo, s.vars, getpenalty(s), s.xmin)
    if safe
        m = meandiag(s.K)
        for i in 1:size(s.K,1)
            if s.K[i,i] ≈ zero(T)
                s.K[i,i] = m
            end
        end
    end
    @timeit to "Solve system of equations" s.u .=  try 
        if T === newT
            Symmetric(K) \ f
        else
            Symmetric(newT.(K)) \ newT.(f)
        end
    catch
        T(NaN)
    end

    nothing
end
