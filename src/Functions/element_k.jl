struct ElementStiffnessMatrix{T<:Real,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Ke::M
end
Base.length(x::ElementStiffnessMatrix) = length(x.x)
Base.size(x::ElementStiffnessMatrix, i...) = size(x.x, i...)
Base.getindex(x::ElementStiffnessMatrix, i...) = x.x[i...]
Base.:*(x::ElementStiffnessMatrix, y) = ElementStiffnessMatrix(x.x * y)

mutable struct ElementK{
    T,
    Ts<:AbstractDisplacementSolver,
    TK1<:AbstractVector{<:AbstractMatrix{T}},
    TK2<:AbstractVector{<:AbstractMatrix{T}},
    Tp<:AbstractPenalty{T},
} <: AbstractFunction{T}
    solver::Ts
    Kes::TK1
    Kes_0::TK2 # un-interpolated
    penalty::Tp
    xmin::T
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ElementK)
    return println("TopOpt element stiffness matrix construction function")
end

function ElementK(solver::AbstractDisplacementSolver)
    @unpack elementinfo = solver
    dh = solver.problem.ch.dh
    penalty = getpenalty(solver)
    xmin = solver.xmin
    solver.vars = ones(getncells(dh.grid))
    # trigger Ke construction
    solver()
    Kes_solver = solver.elementinfo.Kes

    _Ke1 = rawmatrix(Kes_solver[1])
    mat_type = _Ke1 isa Symmetric ? typeof(_Ke1.data) : typeof(_Ke1)
    Kes = mat_type[]
    Kes_0 = mat_type[]
    for (cellidx, _) in enumerate(CellIterator(dh))
        _Ke = rawmatrix(Kes_solver[cellidx])
        Ke0 = _Ke isa Symmetric ? _Ke.data : _Ke
        Ke = similar(Ke0)
        push!(Kes_0, Ke0)
        push!(Kes, Ke)
    end

    return ElementK(solver, Kes, Kes_0, penalty, xmin)
end

function (ek::ElementK)(xe::Number, ci::Integer)
    @unpack xmin, Kes_0, penalty = ek
    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(xe), xmin)
    else
        px = penalty(density(xe), xmin)
    end
    return px * Kes_0[ci]
end

function (ek::ElementK{T})(x::PseudoDensities) where {T}
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x)
    for ci in 1:length(x)
        Kes[ci] = ek(x.x[ci], ci)
    end
    return copy(Kes)
end

function ChainRulesCore.rrule(ek::ElementK, x::PseudoDensities)
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x.x)
    Kes = ek(x)

    """
    g(F(x)), where F = ElementK

    Want: dg/dK_e_ij -> dg/dx_e

    dg/dx_e = sum_i'j' dg/dK_e_i'j' * dK_e_i'j'/dx_e
            = Delta[e][i,j] * dK_e_i'j'/dx_e
    """
    function pullback_fn(Δ)
        Δx = similar(x.x)
        for ci in 1:length(x.x)
            ek_cell_fn = xe -> vec(ek(xe, ci))
            jac_cell = ForwardDiff.derivative(ek_cell_fn, x.x[ci])
            Δx[ci] = jac_cell' * vec(Δ[ci])
        end
        return Tangent{typeof(ek)}(;
            solver=NoTangent(),
            Kes=Δ,
            Kes_0=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
        ),
        Tangent{typeof(x)}(; x=Δx)
    end
    return Kes, pullback_fn
end
