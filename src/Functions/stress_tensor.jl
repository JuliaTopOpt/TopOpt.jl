"""
    StressTensor(solver::AbstractFEASolver)

Element-wise microscopic stress tensor. Computes the symmetric stress tensor
for each element from the nodal displacements using the base Young's modulus.
Call as `σ = σf(u)` where `u` is the displacement vector (e.g. from
`Displacement`). Returns a vector of symmetric matrices, one per element.
"""
struct StressTensor{T,Tp,Ts,Tc1,Tc2} <: AbstractFunction{T}
    problem::Tp
    solver::Ts
    global_dofs::Vector{Int}
    cellvalues::Tc1
    cells::Tc2
    _::T
end
function StressTensor(solver)
    problem = solver.problem
    # StressTensor is only valid for structural (LinearElasticity) problems
    @assert problem isa StiffnessTopOptProblem "StressTensor can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(problem))"
    dh = problem.ch.dh
    n = ndofs_per_cell(dh)
    global_dofs = zeros(Int, n)
    cellvalues = solver.elementinfo.cellvalues
    return StressTensor(
        problem, solver, global_dofs, cellvalues, collect(CellIterator(dh)), 0.0
    )
end

function Ferrite.reinit!(s::StressTensor, cellidx)
    reinit!(s.cellvalues, s.cells[cellidx])
    celldofs!(s.global_dofs, s.problem.ch.dh, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::StressTensor, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function (f::StressTensor)(dofs::DisplacementResult)
    return map(1:length(f.cells)) do cellidx
        cf = f[cellidx]
        return cf(dofs)
    end
end

"""
    ElementStressTensor(solver::AbstractFEASolver)

Element stress tensor operator that also stores per-element metadata for use
in stress-constrained optimization and ML applications.
"""
struct ElementStressTensor{T,Ts<:StressTensor{T},Tc1,Tc2} <: AbstractFunction{T}
    stress_tensor::Ts
    cell::Tc1
    cellidx::Tc2
end
function Base.getindex(f::StressTensor{T}, cellidx) where {T}
    reinit!(f, cellidx)
    return ElementStressTensor(f, f.cells[cellidx], cellidx)
end

function Ferrite.reinit!(s::ElementStressTensor, cellidx)
    reinit!(s.stress_tensor, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::ElementStressTensor, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function (f::ElementStressTensor)(u::DisplacementResult; element_dofs=false)
    st = f.stress_tensor
    reinit!(f, f.cellidx)
    if element_dofs
        return _element_stress_tensor(f, u)
    else
        return _element_stress_tensor(f, DisplacementResult(u.u[copy(st.global_dofs)]))
    end
end

function _element_stress_tensor(f::ElementStressTensor, u::DisplacementResult)
    st = f.stress_tensor
    cellu = u.u
    n_basefuncs = getnbasefunctions(st.cellvalues)
    n_quad = getnquadpoints(st.cellvalues)
    dim = TopOptProblems.getdim(st.problem)
    V = sum(st.cellvalues.detJdV)
    return sum(
        map(1:n_quad) do q_point
            dΩ = getdetJdV(st.cellvalues, q_point)
            sum(
                map(1:n_basefuncs) do a
                    _u = cellu[dim * (a - 1) .+ (1:dim)]
                    return tensor_kernel(f, q_point, a)(DisplacementResult(_u))
                end,
            ) * dΩ
        end,
    ) ./ V
end

function ChainRulesCore.rrule(
    ::typeof(_element_stress_tensor), f::ElementStressTensor, u::DisplacementResult
)
    J = ForwardDiff.jacobian(
        vec ∘ (u -> _element_stress_tensor(f, DisplacementResult(u))), u.u
    )
    return _element_stress_tensor(f, u),
    Δ -> begin
        NoTangent(), NoTangent(), Tangent{typeof(u)}(; u=J' * vec(Δ))
    end
end

struct ElementStressTensorKernel{T,Tc} <: AbstractFunction{T}
    E::T
    ν::T
    q_point::Int
    a::Int
    cellvalues::Tc
    dim::Int
end
function (f::ElementStressTensorKernel)(u::DisplacementResult)
    @unpack E, ν, q_point, a, cellvalues, dim = f
    ∇ϕ = Vector(shape_gradient(cellvalues, q_point, a))
    ϵ = (u.u .* ∇ϕ' .+ ∇ϕ .* u.u') ./ 2
    # 3D isotropic (plane strain) constitutive law, consistent with the
    # stiffness matrix assembly:
    #   σ_ij = λ * ε_kk * δ_ij + 2μ * ε_ij
    #   λ = E*ν/((1+ν)*(1-2ν)),  2μ = E/(1+ν)
    λ = E * ν / ((1 + ν) * (1 - 2 * ν))
    two_mu = E / (1 + ν)
    if dim == 2
        # Plane strain: σzz = λ*(εxx + εyy) ≠ 0, so we must return the full
        # 3×3 tensor for the von Mises formula to be correct.
        tr_ϵ = ϵ[1, 1] + ϵ[2, 2]
        σ = zeros(eltype(u.u), 3, 3)
        σ[1, 1] = λ * tr_ϵ + two_mu * ϵ[1, 1]
        σ[2, 2] = λ * tr_ϵ + two_mu * ϵ[2, 2]
        σ[3, 3] = λ * tr_ϵ
        σ[1, 2] = σ[2, 1] = two_mu * ϵ[1, 2]
        return σ
    else
        return λ * sum(diag(ϵ)) * I + two_mu * ϵ
    end
end
function ChainRulesCore.rrule(f::ElementStressTensorKernel, u::DisplacementResult)
    v, (∇,) = AD.value_and_jacobian(
        AD.ForwardDiffBackend(), u -> vec(f(DisplacementResult(u))), u.u
    )
    out_dim = f.dim == 2 ? 3 : f.dim
    return reshape(v, out_dim, out_dim), Δ -> (NoTangent(), Tangent{typeof(u)}(; u=∇' * vec(Δ)))
end

function tensor_kernel(f::StressTensor, quad, basef)
    return ElementStressTensorKernel(
        f.problem.E,
        f.problem.ν,
        quad,
        basef,
        f.cellvalues,
        TopOptProblems.getdim(f.problem),
    )
end
function tensor_kernel(f::ElementStressTensor, quad, basef)
    return tensor_kernel(f.stress_tensor, quad, basef)
end

function von_mises(σ::AbstractMatrix)
    if size(σ, 1) == 3
        t1 = ((σ[1, 1] - σ[2, 2])^2 + (σ[2, 2] - σ[3, 3])^2 + (σ[3, 3] - σ[1, 1])^2) / 2
        t2 = 3 * (σ[1, 2]^2 + σ[2, 3]^2 + σ[3, 1]^2)
    else
        throw(ArgumentError("Unsupported stress tensor type. Expected 3×3 (plane strain 2D or full 3D)."))
    end
    return sqrt(t1 + t2)
end

"""
    von_mises_stress_function(solver::AbstractFEASolver)

Return a function that computes the element-wise microscopic von Mises stress
from the design. Applies the penalty and interpolation, solves the FEA system,
and computes the von Mises stress for each element using the base Young's
modulus.

Call as `σv = σvf(PseudoDensities(x))`. Returns a vector of von Mises stress
values, one per element.
"""
function von_mises_stress_function(solver::AbstractFEASolver)
    st = StressTensor(solver)
    dp = Displacement(solver)
    return x -> von_mises.(st(dp(x)))
end
