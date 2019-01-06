struct MMAModel{T, TV<:AbstractVector{T}, TC<:AbstractVector{<:Function}}
    dim::Int
    objective::Function
    ineq_constraints::TC
    box_max::TV
    box_min::TV
    # Trace flags
    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool
    # Stopping criteria
    maxiter::Base.RefValue{Int}
    ftol::Base.RefValue{T}
    xtol::Base.RefValue{T}
    grtol::Base.RefValue{T}
end
GPUUtils.whichdevice(m::MMAModel) = whichdevice(m.box_max)

dim(m::MMAModel) = m.dim
min(m::MMAModel, i::Integer) = m.box_min[i]
max(m::MMAModel, i::Integer) = m.box_max[i]
min(m::MMAModel)= m.box_max
max(m::MMAModel) = m.box_min
objective(m::MMAModel) = m.objective
constraints(m::MMAModel) = m.ineq_constraints
constraint(m::MMAModel, i::Integer) = m.ineq_constraints[i]

eval_objective(m, x::AbstractVector{T}) where {T} = eval_objective(m, x, T[])
eval_objective(m, x, ∇g) = eval_objective(whichdevice(objective(m)), m, x, ∇g)
eval_objective(::CPU, m, x::Vector{T}, ∇g) where {T} = T(m.objective(x, ∇g))
eval_objective(::GPU, m, x::CuVector{T}, ∇g) where {T} = T(m.objective(x, ∇g))
function eval_objective(::GPU, m, x::Vector{T}, ∇g) where {T}
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
eval_constraint(::CPU, m, i, x::Vector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))
eval_constraint(::GPU, m, i, x::CuVector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))
function eval_constraint(::GPU, m, i, x::Vector, ∇g)
    x_gpu = CuArray(x)
    ∇g_gpu = CuArray(∇g)
    constr = T(constraint(m, i)(x, ∇g))
    copyto!(∇g, ∇g_gpu)
    return constr
end
function eval_constraint(::CPU, m, i, x::CuVector, ∇g)
    error("Optimization on the GPU with the constraint evaluation on the CPU is weird!")
end

ftol(m) = m.ftol[]
xtol(m) = m.xtol[]
grtol(m) = m.grtol[]
ftol!(m, v) = m.ftol[] = v
xtol!(m, v) = m.xtol[] = v
grtol!(m, v) = m.grtol[] = v

MMAModel(args...; kwargs...) = MMAModel{CPU}(args...; kwargs...)
MMAModel{T}(args...; kwargs...) where T = MMAModel(T(), args...; kwargs...) 
MMAModel(::CPU, args...; kwargs...) = MMAModel{Float64, Vector{Float64}, Vector{Function}}(args...; kwargs...)
MMAModel(::GPU, args...; kwargs...) = MMAModel{Float64, CuVector{Float64}, Vector{Function}}(args...; kwargs...)

function MMAModel{T, TV, TC}(dim,
                  objective::Function;
                  maxiter = 200,
                  xtol = eps(T),
                  ftol = sqrt(eps(T)),
                  grtol = sqrt(eps(T)),
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false) where {T, TV, TC}

    mins = ninfsof(TV, dim)
    maxs = infsof(TV, dim)
    MMAModel{T, TV, TC}(dim, objective, Function[],
             mins, maxs, store_trace, show_trace, extended_trace,
             Ref(maxiter), Ref(T(ftol)), Ref(T(xtol)), Ref(T(grtol)))
end

# Box constraints
function box!(m::MMAModel, i::Integer, minb::T, maxb::T) where {T}
    if !(1 <= i <= dim(m))
        throw(ArgumentError("box constraint need to applied to an existing variable"))
    end
    m.box_min[i] = minb
    m.box_max[i] = maxb
end

function box!(m::MMAModel, minb::T, maxb::T) where {T}
    nv = dim(m)
    m.box_min[1:nv] .= minb
    m.box_max[1:nv] .= maxb
end

function box!(m::MMAModel, minbs::AbstractVector{T}, maxbs::AbstractVector{T}) where {T}
    if (length(minbs) != dim(m)) || (length(minbs) != dim(m))
        throw(ArgumentError("box constraint vector must have same size as problem dimension"))
    end
    nv = dim(m)
    map!(identity, m.box_min, minbs)
    map!(identity, m.box_max, maxbs)
end

function ineq_constraint!(m::MMAModel, f::Function)
    push!(m.ineq_constraints, f)
end

function ineq_constraint!(m::MMAModel, fs::Vector{Function})
    for f in fs
        push!(m.ineq_constraints, f)
    end
end
