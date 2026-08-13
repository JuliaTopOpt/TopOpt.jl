"""
    BlockCompliance(solver, scenarios)

Per-load-case compliance vector for multi-load problems. Returns a vector of
compliance values, one per load case.
"""
mutable struct BlockComplianceFun{
    T,TC<:ComplianceFun{T},TM,TS,Tr<:AbstractVector{T},Tv<:AbstractVector{T}
} <: AbstractFunction{T}
    compliance::TC
    method::TM
    F::TS
    raw_val::Tr
    val::Tv
    fevals::Int
    decay::T
end
function BlockComplianceFun(
    problem::MultiLoad,
    solver::AbstractFEASolver;
    method=:exact,
    sample_once=true,
    nv=nothing,
    V=nothing,
    sample_method=:hutch,
    decay=1.0,
    kwargs...,
)
    # BlockCompliance is only valid for structural (LinearElasticity) problems
    solver.problem isa StiffnessTopOptProblem || throw(
        ArgumentError(
            "BlockComplianceFun can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(solver.problem))",
        ),
    )
    comp = ComplianceFun(solver; kwargs...)
    if method == :exact
        method = ExactDiagonal(problem.F, length(comp.grad))
    elseif method == :exact_svd
        method = ExactSVDDiagonal(problem.F, length(comp.grad))
    else
        if sample_method isa Symbol
            sample_method = sample_method == :hadamard ? hadamard! : hutch_rand!
        end
        if V === nothing
            nv = nv === nothing ? 1 : nv
            method = DiagonalEstimation(
                problem.F, nv, length(comp.grad), sample_once, sample_method
            )
        else
            nv = nv === nothing ? size(V, 2) : nv
            method = DiagonalEstimation(
                problem.F, view(V, :, 1:nv), length(comp.grad), sample_once, sample_method
            )
        end
    end
    val = similar(comp.grad, size(problem.F, 2))
    val .= 0
    raw_val = copy(val)
    return BlockComplianceFun(comp, method, problem.F, raw_val, val, 0, decay)
end
@forward_property BlockComplianceFun compliance

function (bc::BlockComplianceFun{T})(x::PseudoDensities) where {T}
    @unpack compliance, method, raw_val, val, decay = bc
    @unpack solver = compliance
    solver.vars .= x.x
    penalty = getpenalty(bc)
    bc.fevals += 1
    setpenalty!(solver, penalty.p)
    compute_block_compliance(bc, method) # Modifies raw_val
    val .= raw_val
    return val
end

function ChainRulesCore.rrule(bc::BlockComplianceFun, x::PseudoDensities)
    return bc(x), Δ -> begin
        Δ = ChainRulesCore.unthunk(Δ)
        @assert Nonconvex.NonconvexCore.getdim(bc) == length(Δ)
        newΔ = similar(Δ, length(x))
        newΔ .= 0
        @unpack compliance = bc
        @unpack solver = compliance
        @unpack elementinfo = solver
        w = Δ
        compute_jtvp!_bc(newΔ, bc, bc.method, w)
        return (nothing, Tangent{typeof(x)}(; x=newΔ))
    end
end

Nonconvex.NonconvexCore.getdim(f::BlockComplianceFun) = length(f.val)
Utilities.getpenalty(c::BlockComplianceFun) = getpenalty(getsolver(c.compliance))

function compute_block_compliance(ec::BlockComplianceFun, m::ExactDiagonal)
    return compute_exact_bc(ec, m.F, m.Y)
end
function compute_exact_bc(bc, F, Y)
    @unpack compliance, raw_val = bc
    @unpack solver = compliance
    solver(; rhs=F, lhs=Y, assemble_f=false)
    raw_val .= vec(sum(F .* Y; dims=1))
    return raw_val
end
function compute_jtvp!_bc(out, bc, method::ExactDiagonal, w)
    @unpack Y, temp = method
    @unpack solver = bc.compliance
    out .= 0
    for i in axes(Y, 2)
        temp .= 0
        if w[i] != 0
            @views compute_inner(temp, Y[:, i], Y[:, i], solver)
            out .+= w[i] .* temp
        end
    end
    return out
end

function compute_block_compliance(bc::BlockComplianceFun, m::ExactSVDDiagonal)
    return compute_exact_svd_bc(bc, m.F, m.US, m.V, m.Q, m.Y)
end
function compute_exact_svd_bc(bc, F, US, V, Q, Y)
    @unpack compliance, raw_val = bc
    @unpack solver = compliance
    raw_val .= 0
    solver(; assemble_f=false, rhs=US, lhs=Q)
    for i in eachindex(raw_val)
        raw_val[i] = (F[:, i]' * Q) * V[i, :]
    end
    return raw_val
end
function compute_jtvp!_bc(out, bc, method::ExactSVDDiagonal, w)
    @unpack US, V, Q, temp = method
    @unpack solver = bc.compliance
    X = V' * Diagonal(w) * V
    ns = size(US, 2)
    out .= 0
    for j in 1:ns, i in j:ns
        if X[i, j] != 0
            temp .= 0
            @views compute_inner(temp, Q[:, i], Q[:, j], solver)
            if i != j
                out .+= 2 * X[i, j] .* temp
            else
                out .+= X[i, j] .* temp
            end
        end
    end
    return out
end

function compute_block_compliance(bc::BlockComplianceFun, ap::DiagonalEstimation)
    return compute_approx_bc(bc, ap.F, ap.V, ap.Y)
end
function compute_approx_bc(bc, F, V, Y)
    @unpack compliance, raw_val = bc
    @unpack solver = compliance
    nv = size(V, 2)
    raw_val .= 0
    bc.method.sample_once || bc.method.sample_method(V)
    for i in 1:nv
        @views mul!(solver.rhs, F, V[:, i])
        solver(; assemble_f=false, reuse_fact=(i > 1))
        invKFv = solver.lhs
        Y[:, i] .= invKFv
        temp = F' * invKFv
        @views raw_val .+= V[:, i] .* temp
    end
    raw_val ./= nv
    return raw_val
end
function compute_jtvp!_bc(out, bc, method::DiagonalEstimation, w)
    @unpack solver = bc.compliance
    @unpack F, V, Y, Q, temp = method
    nv = size(V, 2)
    out .= 0
    for i in 1:nv
        temp .= 0
        #q_i = K^-1 F (w .* v_i)
        @views mul!(solver.rhs, F, w .* V[:, i])
        solver(; assemble_f=false, reuse_fact=(i > 1))
        Q[:, i] = solver.lhs
        #<q_i, dK/dx_e, y_i>
        @views compute_inner(temp, Q[:, i], Y[:, i], solver)
        out .+= temp
    end
    out ./= nv
    return out
end
