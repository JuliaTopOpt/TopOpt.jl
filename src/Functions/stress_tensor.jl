"""
    StressTensorFun(solver::AbstractFEASolver)

Element-wise microscopic stress tensor. Computes the symmetric stress tensor
for each element from the nodal displacements using the base Young's modulus.
Call as `σ = σf(u)` where `u` is the displacement vector (e.g. from
`DisplacementFun`). Returns a vector of symmetric matrices, one per element.

For 2D problems the stress is computed with the plane-strain constitutive
law (σzz = λ·(εxx + εyy) ≠ 0), consistent with the stiffness matrix assembly.
The returned tensor is 3×3 for 2D problems and 3×3 for 3D problems. See
[Bathe1996](@cite) §4.2 for the plane-strain formulation and
[DuysinxBendsøe1998](@cite) for stress-constrained topology optimization.
"""
struct StressTensorFun{T,Tp,Ts,Tc1,Tc2} <: AbstractFunction{T}
    problem::Tp
    solver::Ts
    global_dofs::Vector{Int}
    cellvalues::Tc1
    cells::Tc2
    _::T
end
function StressTensorFun(solver)
    problem = solver.problem
    # StressTensorFun is only valid for structural (LinearElasticity) problems
    problem isa StiffnessTopOptProblem || throw(
        ArgumentError(
            "StressTensorFun can only be used with StiffnessTopOptProblem (structural mechanics). Got $(typeof(problem))",
        ),
    )
    dh = problem.ch.dh
    n = ndofs_per_cell(dh)
    global_dofs = zeros(Int, n)
    cellvalues = solver.elementinfo.cellvalues
    return StressTensorFun(
        problem, solver, global_dofs, cellvalues, collect(CellIterator(dh)), 0.0
    )
end

function Ferrite.reinit!(s::StressTensorFun, cellidx)
    reinit!(s.cellvalues, s.cells[cellidx])
    celldofs!(s.global_dofs, s.problem.ch.dh, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::StressTensorFun, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function (f::StressTensorFun)(dofs::DisplacementResult)
    return map(eachindex(f.cells)) do cellidx
        cf = f[cellidx]
        return cf(dofs)
    end
end

"""
    ElementStressTensorFun(solver::AbstractFEASolver)

Element stress tensor operator that also stores per-element metadata for use
in stress-constrained optimization and ML applications.
"""
struct ElementStressTensorFun{T,Ts<:StressTensorFun{T},Tc1,Tc2} <: AbstractFunction{T}
    stress_tensor::Ts
    cell::Tc1
    cellidx::Tc2
end
function Base.getindex(f::StressTensorFun, cellidx)
    reinit!(f, cellidx)
    return ElementStressTensorFun(f, f.cells[cellidx], cellidx)
end

function Ferrite.reinit!(s::ElementStressTensorFun, cellidx)
    reinit!(s.stress_tensor, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::ElementStressTensorFun, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function (f::ElementStressTensorFun)(u::DisplacementResult; element_dofs=false)
    st = f.stress_tensor
    reinit!(f, f.cellidx)
    if element_dofs
        return _element_stress_tensor(f, u)
    else
        return _element_stress_tensor(f, DisplacementResult(u.u[copy(st.global_dofs)]))
    end
end

function _element_stress_tensor(f::ElementStressTensorFun, u::DisplacementResult)
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
    ::typeof(_element_stress_tensor), f::ElementStressTensorFun, u::DisplacementResult
)
    J = ForwardDiff.jacobian(
        vec ∘ (u -> _element_stress_tensor(f, DisplacementResult(u))), u.u
    )
    return _element_stress_tensor(f, u),
    Δ -> begin
        Δ = ChainRulesCore.unthunk(Δ)
        NoTangent(), NoTangent(), Tangent{typeof(u)}(; u=J' * vec(Δ))
    end
end

struct ElementStressTensorKernelFun{T,Tc} <: AbstractFunction{T}
    E::T
    ν::T
    q_point::Int
    a::Int
    cellvalues::Tc
    dim::Int
end
function (f::ElementStressTensorKernelFun)(u::DisplacementResult)
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
function ChainRulesCore.rrule(f::ElementStressTensorKernelFun, u::DisplacementResult)
    v, ∇ = DI.value_and_jacobian(
        u -> vec(f(DisplacementResult(u))), DI.AutoForwardDiff(), u.u
    )
    out_dim = f.dim == 2 ? 3 : f.dim
    return reshape(v, out_dim, out_dim),
    Δ -> (NoTangent(), Tangent{typeof(u)}(; u=∇' * vec(ChainRulesCore.unthunk(Δ))))
end

function tensor_kernel(f::StressTensorFun, quad, basef)
    return ElementStressTensorKernelFun(
        f.problem.E,
        f.problem.ν,
        quad,
        basef,
        f.cellvalues,
        TopOptProblems.getdim(f.problem),
    )
end
function tensor_kernel(f::ElementStressTensorFun, quad, basef)
    return tensor_kernel(f.stress_tensor, quad, basef)
end

"""
    von_mises(σ::AbstractMatrix)

Compute the von Mises equivalent stress from a 3×3 symmetric stress tensor.
For 2D problems, the stress tensor from `StressTensorFun` / `ElementStressTensorKernel`
includes the plane-strain out-of-plane component `σzz = λ·(εxx + εyy)`, so
the full 3D von Mises formula is used:

```
σ_vm = sqrt(½[(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)²] + 3(σxy² + σyz² + σzx²))
```

This is consistent with the plane-strain constitutive law used in the
stiffness matrix assembly. Using the plane-stress von Mises formula (which
assumes `σzz = 0`) with a plane-strain stress tensor would overestimate the
von Mises stress by up to ~33% for typical Poisson's ratios. See
[Bathe1996](@cite) §4.2 and [DuysinxBendsøe1998](@cite).
"""
function von_mises(σ::AbstractMatrix)
    if size(σ, 1) == 3
        t1 = ((σ[1, 1] - σ[2, 2])^2 + (σ[2, 2] - σ[3, 3])^2 + (σ[3, 3] - σ[1, 1])^2) / 2
        t2 = 3 * (σ[1, 2]^2 + σ[2, 3]^2 + σ[3, 1]^2)
    else
        throw(
            ArgumentError(
                "Unsupported stress tensor type. Expected 3×3 (plane strain 2D or full 3D)."
            ),
        )
    end
    return sqrt(t1 + t2)
end

"""
    von_mises_stress_function(solver::AbstractFEASolver; stress_exponent=0)

Return a function that computes the element-wise von Mises stress from the
design. Applies the penalty and interpolation, solves the FEA system, and
computes the von Mises stress for each element using the base Young's modulus.

Call as `σv = σvf(PseudoDensities(x))`. Returns a vector of von Mises stress
values, one per element.

With `stress_exponent = 0` (default) the returned stress is the microscopic
stress `σ = C_0 : ε` of [DuysinxBendsøe1998](@cite), which is finite at zero
density and therefore exhibits the stress-singularity phenomenon: optimal
designs with vanishing members are unreachable for gradient-based optimizers.
With `stress_exponent = q > 0` the returned stress is the relaxed (penalized)
stress `σ̃_e = ρ_e^q σ_e`, where `ρ_e` is the physical density
(`xmin + (1 - xmin) x_e`). The relaxation acts on the physical density, not
the penalized stiffness density, so it is independent of the SIMP penalty and
of the `PENALTY_BEFORE_INTERPOLATION` preference; the `xmin` floor keeps `ρ_e^q`
finite and differentiable at zero density. Because `σ̃_e → 0` as `ρ_e → 0`,
low-density elements automatically satisfy any stress bound, which removes the
singular optima. `q = 0.5` with SIMP stiffness exponent `p = 3` is the choice of
[Le2010](@cite); the equivalent constraint-side formulation is the qp-approach
of [Bruggi2008](@cite) with relaxation `ε_qp = p - q`. See also
[ChengGuo1997](@cite) for the alternative ε-relaxation (available here as
[`epsilon_relaxed`](@ref)) and [Verbart2017](@cite) for a review of both.
"""
function von_mises_stress_function(solver::AbstractFEASolver; stress_exponent=0)
    st = StressTensorFun(solver)
    dp = DisplacementFun(solver)
    if iszero(stress_exponent)
        return x -> von_mises.(st(dp(x)))
    end
    xmin = solver.xmin
    return x -> begin
        σv = von_mises.(st(dp(x)))
        ρ = density.(x.x, xmin)
        return ρ .^ stress_exponent .* σv
    end
end
