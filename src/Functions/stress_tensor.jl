@params struct StressTensor{T} <: AbstractFunction{T}
    problem
    solver
    global_dofs::Vector{Int}
    cellvalues
    cells
    _::T
end
function StressTensor(solver)
    problem = solver.problem
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

function (f::StressTensor)(dofs)
    return map(1:length(f.cells)) do cellidx
        cf = f[cellidx]
        return cf(dofs)
    end
end

@params struct ElementStressTensor{T} <: AbstractFunction{T}
    stress_tensor::StressTensor{T}
    cell
    cellidx
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

function (f::ElementStressTensor)(u; element_dofs=false)
    st = f.stress_tensor
    reinit!(f, f.cellidx)
    if element_dofs
        cellu = u
    else
        cellu = u[copy(st.global_dofs)]
    end
    n_basefuncs = getnbasefunctions(st.cellvalues)
    n_quad = getnquadpoints(st.cellvalues)
    dim = TopOptProblems.getdim(st.problem)
    return sum(
        map(1:n_basefuncs, 1:n_quad) do a, q_point
            _u = cellu[dim * (a - 1) .+ (1:dim)]
            return tensor_kernel(f, q_point, a)(_u)
        end,
    )
end

@params struct ElementStressTensorKernel{T} <: AbstractFunction{T}
    E::T
    ν::T
    q_point::Int
    a::Int
    cellvalues
    dim::Int
end
function (f::ElementStressTensorKernel)(u)
    @unpack E, ν, q_point, a, cellvalues = f
    ∇ϕ = Vector(shape_gradient(cellvalues, q_point, a))
    ϵ = (u .* ∇ϕ' .+ ∇ϕ .* u') ./ 2
    c1 = E * ν / (1 - ν^2) * sum(diag(ϵ))
    c2 = E * ν * (1 + ν)
    return c1 * I + c2 * ϵ
end
function ChainRulesCore.rrule(f::ElementStressTensorKernel, x::AbstractVector)
    v, (∇,) = AD.value_and_jacobian(AD.ForwardDiffBackend(), x -> vec(f(x)), x)
    return reshape(v, f.dim, f.dim), Δ -> (NoTangent(), ∇' * vec(Δ))
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
    if size(σ, 1) == 2
        t1 = σ[1, 1]^2 - σ[1, 1] * σ[2, 2] + σ[2, 2]^2
        t2 = 3 * σ[1, 2]^2
    elseif size(σ, 1) == 3
        t1 = ((σ[1, 1] - σ[2, 2])^2 + (σ[2, 2] - σ[3, 3])^2 + (σ[3, 3] - σ[1, 1])^2) / 2
        t2 = 3 * (σ[1, 2]^2 + σ[2, 3]^2 + σ[3, 1]^2)
    else
        throw(ArgumentError("Unsupported stress tensor type."))
    end
    return sqrt(t1 + t2)
end

function von_mises_stress_function(solver::AbstractFEASolver)
    st = StressTensor(solver)
    dp = Displacement(solver)
    return x -> von_mises.(st(dp(x)))
end
