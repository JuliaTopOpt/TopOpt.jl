@params mutable struct LinAggregation{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
    fval::AbstractVector{T}
    weights::AbstractVector{T}
    grad::AbstractVector{T}
    fevals::Int
    maxfevals::Int
end

@inline function Base.getproperty(vf::LinAggregation, f::Symbol)
    f === :reuse && return vf.f.reuse
    f === :solver && return vf.f.solver
    return getfield(vf, f)
end
@inline function Base.setproperty!(vf::LinAggregation, f::Symbol, v)
    f === :reuse && return setproperty!(vf.f, f, v)
    return setfield!(vf, f, v)
end

function LinAggregation(f::AbstractFunction{T}, weights::AbstractVector{T}; maxfevals = 10^8) where {T}
    grad = similar(f.grad)
    fval = similar(grad, TopOpt.dim(f))
    return LinAggregation(f, fval, weights, grad, 0, maxfevals)
end
# To be defined efficiently for every block constraint
function (v::LinAggregation{T})(x, grad = v.grad) where {T}
    @assert length(v.weights) == length(v.fval) == 1
    v.fevals += 1
    v.fval[1] = v.f(x)
    grad .= v.f.grad .* v.weights[1]
    val = v.fval[1] * v.weights[1]
    if grad !== v.grad
        v.grad .= grad
    end
    return val
end

for TF in (:QuadAggregation, :QuadMaxAggregation)
    @eval begin
        @params mutable struct $TF{T} <: AbstractFunction{T}
            f::AbstractFunction{T}
            fval::AbstractVector{T}
            weight::T
            grad::AbstractVector{T}
            fevals::Int
            maxfevals::Int
        end

        @inline function Base.getproperty(vf::$TF, f::Symbol)
            f === :reuse && return vf.f.reuse
            f === :solver && return vf.f.solver
            return getfield(vf, f)
        end
        @inline function Base.setproperty!(vf::$TF, f::Symbol, v)
            f === :reuse && return setproperty!(vf.f, f, v)
            return setfield!(vf, f, v)
        end

        function $TF(f::AbstractFunction{T}, weight::T; maxfevals = 10^8) where {T}
            grad = similar(f.grad)
            fval = similar(grad, TopOpt.dim(f))
            return $TF(f, fval, weight, grad, 0, maxfevals)
        end
        # To be defined efficiently for every block constraint
        function (v::$TF{T})(x, grad = v.grad) where {T}
            @assert TopOpt.dim(v.f) == length(v.fval) == 1
            v.fevals += 1
            v.fval[1] = v.f(x)
            if $TF <: QuadAggregation
                val = v.fval[1]
            else
                val = max(v.fval[1], 0)
            end
            grad .= v.f.grad .* v.weight .* 2 * val
            val = v.weight * val^2
            if grad !== v.grad
                v.grad .= grad
            end
            return val
        end
    end
end

for TF in (:LinQuadAggregation, :LinQuadMaxAggregation)
    @eval begin
        @params mutable struct $TF{T} <: AbstractFunction{T}
            f::AbstractFunction{T}
            fval::AbstractVector{T}
            lin_weights::AbstractVector{T}
            quad_weight::T
            grad::AbstractVector{T}
            fevals::Int
            maxfevals::Int
        end

        @inline function Base.getproperty(vf::$TF, f::Symbol)
            f === :reuse && return vf.f.reuse
            f === :solver && return vf.f.solver
            return getfield(vf, f)
        end
        @inline function Base.setproperty!(vf::$TF, f::Symbol, v)
            f === :reuse && return setproperty!(vf.f, f, v)
            return setfield!(vf, f, v)
        end

        function $TF(f::AbstractFunction{T}, lin_weights::AbstractVector{T}, quad_weight::T; maxfevals = 10^8) where {T}
            grad = similar(f.grad)
            fval = similar(grad, TopOpt.dim(f))
            return $TF(f, fval, lin_weights, quad_weight, grad, 0, maxfevals)
        end
        # To be defined efficiently for every block constraint
        function (v::$TF{T})(x, grad = v.grad) where {T}
            @assert TopOpt.dim(v.f) == length(v.fval) == length(v.lin_weights) == 1
            v.fval[1] = v.f(x)
            if $TF <: LinQuadAggregation
                val_quad = v.fval[1]
            else
                val_quad = max(v.fval[1], 0)
            end
            grad .= v.f.grad .* (v.quad_weight * 2 * val_quad .+ v.lin_weights[1])
            val = val_quad^2 * v.quad_weight + v.fval[1] * v.lin_weights[1]
            v.fevals += 1
            if grad !== v.grad
                v.grad .= grad
            end
            return val
        end
    end
end
