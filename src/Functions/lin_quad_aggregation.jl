@params mutable struct LinAggregation{T} <: AbstractFunction{T}
    f::AbstractFunction{T}
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
    return LinAggregation(f, weights, grad, 0, maxfevals)
end
# To be defined efficiently for every block constraint
function (v::LinAggregation{T})(x, grad = v.grad) where {T}
    @assert length(v.weights) == 1
    val = v.f(x)
    grad .= v.f.grad .* v.weights[1]
    val = val * v.weights[1]
    v.fevals += 1
    if grad !== v.grad
        v.grad .= grad
    end
    return val
end

for TF in (:QuadAggregation, :QuadMaxAggregation)
    @eval begin
        @params mutable struct $TF{T} <: AbstractFunction{T}
            f::AbstractFunction{T}
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
            return $TF(f, weight, grad, 0, maxfevals)
        end
        # To be defined efficiently for every block constraint
        function (v::$TF{T})(x, grad = v.grad) where {T}
            if $TF <: QuadAggregation
                val = v.f(x)
            else
                val = max(v.f(x), 0)
            end
            grad .= v.f.grad .* v.weight .* 2 * val
            val = v.weight * val^2
            v.fevals += 1
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
            return $TF(f, lin_weights, quad_weight, grad, 0, maxfevals)
        end
        # To be defined efficiently for every block constraint
        function (v::$TF{T})(x, grad = v.grad) where {T}
            val = v.f(x)
            if $TF <: LinQuadAggregation
                val_quad = val
            else
                val_quad = max(val, 0)
            end
            grad .= v.f.grad .* (v.quad_weight * 2 * val_quad .+ v.lin_weights[1])
            val = val_quad^2 * v.quad_weight + val * v.lin_weights[1]
            v.fevals += 1
            if grad !== v.grad
                v.grad .= grad
            end
            return val
        end
    end
end
