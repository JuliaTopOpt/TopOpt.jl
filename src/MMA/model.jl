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

dim(m::MMAModel) = m.dim
min(m::MMAModel, i::Integer) = m.box_min[i]
max(m::MMAModel, i::Integer) = m.box_max[i]
min(m::MMAModel)= m.box_max
max(m::MMAModel) = m.box_min
objective(m::MMAModel) = m.objective
constraints(m::MMAModel) = m.ineq_constraints
constraint(m::MMAModel, i::Integer) = m.ineq_constraints[i]
eval_objective(m, x::AbstractVector{T}, ∇g) where {T} = T(m.objective(x, ∇g))
eval_objective(m, x::AbstractVector{T}) where {T} = m.objective(x, T[])
eval_constraint(m, i, x::AbstractVector{T}, ∇g) where {T} = T(constraint(m, i)(x, ∇g))
eval_constraint(m, i, x::AbstractVector{T}) where {T} = constraint(m, i)(x, T[])
ftol(m) = m.ftol[]
xtol(m) = m.xtol[]
grtol(m) = m.grtol[]
ftol!(m, v) = m.ftol[] = v
xtol!(m, v) = m.xtol[] = v
grtol!(m, v) = m.grtol[] = v

MMAModel(dim, objective, args...; kwargs...) = MMAModel(whichdevice(objective), dim, objective, args...; kwargs...) 
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
