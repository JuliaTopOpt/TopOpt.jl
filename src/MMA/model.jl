mutable struct Model{T, TV<:AbstractVector{T}, TC<:AbstractVector{<:Function}}
    dim::Int
    objective::Function
    ineq_constraints::TC
    box_max::TV
    box_min::TV
end
GPUUtils.whichdevice(m::Model) = whichdevice(m.box_max)

dim(m::Model) = m.dim
min(m::Model, i::Integer) = m.box_min[i]
max(m::Model, i::Integer) = m.box_max[i]
min(m::Model)= m.box_max
max(m::Model) = m.box_min
objective(m::Model) = m.objective
constraints(m::Model) = m.ineq_constraints
constraint(m::Model, i::Integer) = m.ineq_constraints[i]

eval_objective(m, x::AbstractVector{T}) where {T} = eval_objective(m, x, T[])
eval_objective(m, x, ∇g) = eval_objective(whichdevice(objective(m)), m, x, ∇g)
function eval_objective(::CPU, m, x::AbstractVector{T}, ∇g) where {T}
    return T(m.objective(x, ∇g))
end
eval_objective(::GPU, m, x::GPUVector{T}, ∇g) where {T} = T(m.objective(x, ∇g))
function eval_objective(::GPU, m, x::AbstractVector{T}, ∇g) where {T}
    x_gpu = CuArray(x)
    ∇g_gpu = CuArray(∇g)
    obj = T(m.objective(x_gpu, ∇g_gpu))
    copyto!(∇g, ∇g_gpu)
    return obj
end
function eval_objective(::CPU, m, x::CuVector{T}, ∇g) where {T}
    error("Optimization on the GPU with the objective evaluation on the CPU is weird!")
end

eval_constraint(m, i, x::AbstractVector{T}) where {T} = eval_constraint(m, i, x, T[])
eval_constraint(m, i, x, ∇g) = eval_constraint(whichdevice(constraint(m, i)), m, i, x, ∇g)
eval_constraint(::CPU, m, i, x::AbstractVector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))
eval_constraint(::GPU, m, i, x::GPUVector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))
function eval_constraint(::GPU, m, i, x::AbstractVector{T}, ∇g) where T
    x_gpu = CuArray(x)
    ∇g_gpu = CuArray(∇g)
    constr = T(constraint(m, i)(x_gpu, ∇g_gpu))
    copyto!(∇g, ∇g_gpu)
    return constr
end
function eval_constraint(::CPU, m, i, x::GPUVector, ∇g)
    error("Optimization on the GPU with the constraint evaluation on the CPU is weird!")
end

ftol(m) = m.ftol
xtol(m) = m.xtol
grtol(m) = m.grtol
ftol!(m, v) = m.ftol = v
xtol!(m, v) = m.xtol = v
grtol!(m, v) = m.grtol = v

Model(args...; kwargs...) = Model{CPU}(args...; kwargs...)
Model{T}(args...; kwargs...) where T = Model(T(), args...; kwargs...) 
Model(::CPU, args...; kwargs...) = Model{Float64, Vector{Float64}, Vector{Function}}(args...; kwargs...)
Model(::GPU, args...; kwargs...) = Model{Float64, CuVector{Float64}, Vector{Function}}(args...; kwargs...)

function Model{T, TV, TC}(dim, objective::Function) where {T, TV, TC}
    mins = ninfsof(TV, dim)
    maxs = infsof(TV, dim)
    Model{T, TV, TC}(dim, objective, Function[],
             mins, maxs)
end

# Box constraints
function box!(m::Model, i::Integer, minb::T, maxb::T) where {T}
    if !(1 <= i <= dim(m))
        throw(ArgumentError("box constraint need to applied to an existing variable"))
    end
    m.box_min[i] = minb
    m.box_max[i] = maxb
end

function box!(m::Model, minb::T, maxb::T) where {T}
    nv = dim(m)
    m.box_min[1:nv] .= minb
    m.box_max[1:nv] .= maxb
end

function box!(m::Model, minbs::AbstractVector{T}, maxbs::AbstractVector{T}) where {T}
    if (length(minbs) != dim(m)) || (length(minbs) != dim(m))
        throw(ArgumentError("box constraint vector must have same size as problem dimension"))
    end
    nv = dim(m)
    map!(identity, m.box_min, minbs)
    map!(identity, m.box_max, maxbs)
end

function ineq_constraint!(m::Model, f::Function)
    push!(m.ineq_constraints, f)
end

function ineq_constraint!(m::Model, fs::Vector{Function})
    for f in fs
        push!(m.ineq_constraints, f)
    end
end
