"""
    TemperatureFun(solver::AbstractFEASolver)

Differentiable nodal temperature function for heat transfer problems. Solves
the thermal FEA system and returns the nodal temperature vector, analogous to
[`DisplacementFun`](@ref) for structural mechanics.

Construct with `TemperatureFun(solver)`. Call as `T = tf(PseudoDensities(x))`.

The adjoint-based gradient solves `K λ = Δ` (reusing the factorization) and
computes `dT/dx_e = -dρ_e/dx_e · T_eᵀ K_e λ`, where `K_e` is the element
conductivity matrix.
"""
mutable struct TemperatureFun{
    T,
    Tt<:AbstractVector{T},
    Td<:AbstractVector,
    Ts<:AbstractFEASolver,
    Tg<:AbstractVector{<:Integer},
} <: AbstractFunction{T}
    T::Tt # temperature vector
    dTdx_tmp::Td # directional derivative
    solver::Ts
    global_dofs::Tg
    fevals::Int
    maxfevals::Int
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TemperatureFun)
    return println(io, "TopOpt temperature function")
end

struct TemperatureResult{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    T::A
end

Base.length(t::TemperatureResult) = length(t.T)
Base.size(t::TemperatureResult, i...) = size(t.T, i...)
Base.getindex(t::TemperatureResult, i::Integer...) = t.T[i...]
Base.getindex(t::TemperatureResult, i::CartesianIndex) = t.T[i]
Base.sum(t::TemperatureResult) = sum(t.T)

"""
    TemperatureFun(solver)

Construct the `TemperatureFun` function struct for a heat transfer solver.
"""
function TemperatureFun(solver::AbstractFEASolver; maxfevals=10^8)
    # TemperatureFun is only valid for heat transfer problems
    solver.problem isa HeatTransferTopOptProblem || throw(
        ArgumentError(
            "TemperatureFun can only be used with HeatTransferTopOptProblem. Got $(typeof(solver.problem))",
        ),
    )
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    total_ndof = ndofs(dh)
    u = zeros(T, total_ndof)
    dTdx_tmp = zeros(T, length(solver.vars))
    return TemperatureFun(u, dTdx_tmp, solver, global_dofs, 0, maxfevals)
end

"""
# Arguments
`x` = design variables

# Returns
temperature vector `T`
"""
function (tf::TemperatureFun{T})(x::PseudoDensities) where {T}
    @unpack solver, global_dofs = tf
    tf.fevals += 1
    length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh) || throw(
        DimensionMismatch(
            "TemperatureFun: global_dofs length $(length(global_dofs)) != ndofs_per_cell $(ndofs_per_cell(solver.problem.ch.dh))",
        ),
    )
    solver.vars .= x.x
    solver()
    return TemperatureResult(copy(solver.u))
end

"""
rrule for autodiff.

dT/dx_e = -K⁻¹ * dK/dx_e * T
        = -K⁻¹ * [d(ρ_e)/d(x_e) * K_e * T]
dT/dx_e' * Δ = -d(ρ_e)/d(x_e) * T_e' * K_e * (K⁻¹ * Δ)
"""
function ChainRulesCore.rrule(tf::TemperatureFun, x::PseudoDensities)
    @unpack dTdx_tmp, solver, global_dofs = tf
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    # Forward pass
    u = tf(x)
    return u, Δ -> begin
        Δ = ChainRulesCore.unthunk(Δ)
        if hasproperty(Δ, :T)
            solver.rhs .= Δ.T
        else
            solver.rhs .= Δ
        end
        solver(; reuse_fact=true, assemble_f=false)
        dTdx_tmp .= 0
        for e in eachindex(x.x)
            _, dρe = get_ρ_dρ(x.x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            # Use the full element conductivity matrix (not the BC-zeroed
            # `bcmatrix`) so the prescribed-temperature lift K_fp·T_p enters
            # the gradient when the Dirichlet values are nonzero.
            KeT = rawmatrix(Kes[e]) * u.T[global_dofs]
            dTdx_tmp[e] = -dρe * dot(KeT, solver.lhs[global_dofs])
        end
        return nothing, Tangent{typeof(x)}(; x=dTdx_tmp)
    end
end

"""
    cell_temperature(T, problem::HeatTransferTopOptProblem)

Average a nodal temperature field `T` (a `TemperatureResult` or plain vector)
over each cell's nodes, returning a per-cell temperature vector suitable for
`visualize(problem; cell_colors=...)`.
"""
function cell_temperature(T, problem::HeatTransferTopOptProblem)
    Tvec = T isa TemperatureResult ? T.T : T
    return [mean(Tvec[node] for node in cell.nodes) for cell in getdh(problem).grid.cells]
end

export TemperatureFun, cell_temperature
