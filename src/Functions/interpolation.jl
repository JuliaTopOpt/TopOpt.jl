struct MultiMaterialVariables{M <: AbstractMatrix}
    x::M
end
function MultiMaterialVariables(x::AbstractVector, nmats::Int)
    d, r = divrem(length(x), nmats - 1)
    @assert r == 0
    return MultiMaterialVariables(reshape(x, d, nmats - 1))
end
function element_densities(x::PseudoDensities, densities::AbstractVector)
    return x.x * densities
end

function Base.sum(x::MultiMaterialVariables; dims)
    return sum(x.x; dims)
end

struct MaterialInterpolation{T, P}
    Es::Vector{T}
    penalty::P
end
(f::MaterialInterpolation)(x::PseudoDensities) = f(x.x)
(f::MaterialInterpolation)(x::MultiMaterialVariables) = f(x.x)
function (f::MaterialInterpolation)(x::AbstractVector)
    d, r = divrem(length(x), length(f.Es) - 1)
    @assert r == 0
    return f(reshape(x, d, length(f.Es) - 1))
end
function (f::MaterialInterpolation)(x::AbstractMatrix)
    @assert size(x, 2) == length(f.Es) - 1
    y = map(f.penalty, tounit(x)) * f.Es
    return PseudoDensities(y)
end

function Utilities.setpenalty!(interp::MaterialInterpolation, p::Real)
    return Utilities.setpenalty!(interp.penalty, p)
end

function PseudoDensities(x::MultiMaterialVariables)
    return PseudoDensities(tounit(x.x))
end

function tounit(x::AbstractVector)
    n = length(x) + 1
    T = eltype(x)
    stick = one(T)
    y = Vector{T}(undef, n)
    for i in 1:n-1
        xi = x[i]
        z = logistic(xi - log(n-i))
        y[i] = z * stick
        stick *= 1 - z
    end
    y[end] = stick
    return y
end
function tounit(x::Matrix)
    return mapreduce(x -> tounit(x)', vcat, eachrow(x))
end
function ChainRulesCore.rrule(::typeof(tounit), x::Vector)
    return tounit(x), Δ -> (NoTangent(), ForwardDiff.jacobian(tounit, x)' * Δ)
end
function ChainRulesCore.rrule(::typeof(tounit), x::Matrix)
    pb = (x, Δ) -> (ForwardDiff.jacobian(tounit, x)' * Δ)'
    return tounit(x), Δ -> (NoTangent(), mapreduce(pb, vcat, eachrow(x), eachrow(Δ)))
end
