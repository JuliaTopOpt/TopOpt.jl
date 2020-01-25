abstract type AbstractModel{T, TV} end

mutable struct Model{T, TV<:AbstractVector{T}, TC<:AbstractVector{<:Function}} <: AbstractModel{T, TV}
    dim::Int
    objective::Function
    ineq_constraints::TC
    box_max::TV
    box_min::TV
end

dim(m::AbstractModel) = m.dim
min(m::AbstractModel, i::Integer) = m.box_min[i]
max(m::AbstractModel, i::Integer) = m.box_max[i]
min(m::AbstractModel)= m.box_max
max(m::AbstractModel) = m.box_min
objective(m::AbstractModel) = m.objective
constraints(m::AbstractModel) = m.ineq_constraints
constraint(m::AbstractModel, i::Integer) = m.ineq_constraints[i]

eval_objective(m::AbstractModel, x::AbstractVector{T}) where {T} = eval_objective(m, x, T[])
eval_objective(m::AbstractModel, x, ∇g) = eval_objective(whichdevice(objective(m)), m, x, ∇g)
function eval_objective(::CPU, m, x::AbstractVector{T}, ∇g) where {T}
    return T(m.objective(x, ∇g))
end

eval_constraint(m::AbstractModel, i, x::AbstractVector{T}) where {T} = eval_constraint(m, i, x, T[])
eval_constraint(m::AbstractModel, i, x, ∇g) = eval_constraint(whichdevice(constraint(m, i)), m, i, x, ∇g)
eval_constraint(::CPU, m, i, x::AbstractVector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))

ftol(m::AbstractModel) = m.ftol
xtol(m::AbstractModel) = m.xtol
grtol(m::AbstractModel) = m.grtol
ftol!(m::AbstractModel, v) = m.ftol = v
xtol!(m::AbstractModel, v) = m.xtol = v
grtol!(m::AbstractModel, v) = m.grtol = v

Model(args...; kwargs...) = Model{CPU}(args...; kwargs...)
Model{T}(args...; kwargs...) where T = Model(T(), args...; kwargs...) 
Model(::CPU, args...; kwargs...) = Model{Float64, Vector{Float64}, Vector{Function}}(args...; kwargs...)

function Model{T, TV, TC}(dim, objective::Function) where {T, TV, TC}
    mins = ninfsof(TV, dim)
    maxs = infsof(TV, dim)
    Model{T, TV, TC}(dim, objective, Function[],
             mins, maxs)
end

# Box constraints
function box!(m::AbstractModel, i::Integer, minb::T, maxb::T) where {T}
    if !(1 <= i <= dim(m))
        throw(ArgumentError("box constraint need to applied to an existing variable"))
    end
    m.box_min[i] = minb
    m.box_max[i] = maxb
end

function box!(m::AbstractModel, minb::T, maxb::T) where {T}
    nv = dim(m)
    m.box_min[1:nv] .= minb
    m.box_max[1:nv] .= maxb
end

function box!(m::AbstractModel, minbs::AbstractVector{T}, maxbs::AbstractVector{T}) where {T}
    if (length(minbs) != dim(m)) || (length(minbs) != dim(m))
        throw(ArgumentError("box constraint vector must have same size as problem dimension"))
    end
    nv = dim(m)
    map!(identity, m.box_min, minbs)
    map!(identity, m.box_max, maxbs)
end

function ineq_constraint!(m::AbstractModel, f::Function)
    push!(m.ineq_constraints, f)
end

function ineq_constraint!(m::AbstractModel, fs::Vector{Function})
    for f in fs
        push!(m.ineq_constraints, f)
    end
end
