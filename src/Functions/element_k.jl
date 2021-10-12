@params mutable struct ElementK{T} <: AbstractFunction{T}
    solver::AbstractDisplacementSolver
    K_e::AbstractMatrix{T}
    K_e0::AbstractMatrix{T}
    cellidx
    penalty
    xmin
end

function ElementK(solver::AbstractDisplacementSolver, cellidx)
    Es = getE(problem)
    As = getA(problem)
    dh = problem.ch.dh
    @unpack elementinfo = solver
    penalty = getpenalty(solver)
    xmin = solver.xmin

    _Ke = rawmatrix(Kes[cellidx])
    K_e0 = _Ke isa Symmetric ? _Ke.data : _Ke
    T = typeof(K_e0[1,1])
    # K_e = zeros(T, size(K_e0))
    K_e = similar(K_e0)

    return ElementK(problem, K_e, K_e0, cellidx, penalty, xmin)
end

# x_e is scalar variable
function (ek::ElementK{T})(x_e) where {T}
    @unpack xmin, K_e0, K_e = ek
    # ? black, white
    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(x_e), xmin)
    else
        px = penalty(density(x_e), xmin)
    end
    K_e = px * K_e0
    return K_e
end

function ChainRulesCore.rrule(ek::ElementK, x_e)
    val = ek(x_e)
    # TODO vec and de-vec
    jac = ForwardDiff.jacobian(ek, x_e)
    val, Δ -> (NoTangent(), jac' * Δ)
end

