@params mutable struct ElementK{T} <: AbstractFunction{T}
    solver::AbstractDisplacementSolver
    Kes::AbstractVector{<:AbstractMatrix{T}}
    Kes_0::AbstractVector{<:AbstractMatrix{T}} # un-interpolated
    penalty
    xmin
end

function ElementK(solver::AbstractDisplacementSolver)
    @unpack elementinfo = solver
    dh = solver.problem.ch.dh
    penalty = getpenalty(solver)
    xmin = solver.xmin
    T = typeof(rawmatrix(Kes[1])[1,1])

    Kes = []
    Kes_0 = []
    for (cellidx, _) in enumerate(CellIterator(dh))
        _Ke = rawmatrix(Kes[cellidx])
        Ke0 = _Ke isa Symmetric ? _Ke.data : _Ke
        Ke = similar(Ke0)
        push!(Kes_0, Ke0)
        push!(Kes, Ke)
    end

    return ElementK(problem, Kes, Kes_0, penalty, xmin)
end

function (ek::ElementK{T})(xe::T, ci) where {T}
    @unpack xmin, Kes_0 = ek
    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(xe), xmin)
    else
        px = penalty(density(xe), xmin)
    end
    return px * Kes_0[ci]
end

function (ek::ElementK{T})(x::AbstractVector{T}) where {T}
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x)
    for ci in 1:length(x)
        Kes[ci] .= ek(x[ci], ci)
    end
    return copy(Kes)
end

"""
g(F(x)), where F = ElementK

Want: dg/dK_e_ij -> dg/dx_e

dg/dx_e = sum_i'j' dg/dK_e_i'j' * dK_e_i'j'/dx_e
        = Delta[e][i,j] * dK_e_i'j'/dx_e
"""
function ChainRulesCore.rrule(ek::ElementK, x)
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x)
    Kes = ek(x_e)
    function pullback_fn(Δ)
        Δx = similar(x)
        for ci in 1:length(x)
            ek_cell = xe -> vec(ek(xe, ci))
            jac_cell = ForwardDiff.jacobian(ek_cell, x[ci])
            Δx[ci] = jac_cell' * vec(Δ[ci])
        end
        return Tangent{typeof(ek)}(solver = NoTangent(), Kes = Δ, Kes_0 = NoTangent()), Δx
    end
    return val, pullback_fn
end

