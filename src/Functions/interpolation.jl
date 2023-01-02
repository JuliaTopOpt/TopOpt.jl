assert_eq(x1, x2) = @assert x1 == x2
function ChainRulesCore.rrule(::typeof(assert_eq), x1, x2)
    assert_eq(x1, x2), _ -> (NoTangent(), NoTangent(), NoTangent())
end

struct MultiMaterialVariables{M<:AbstractMatrix}
    x::M
end
function MultiMaterialVariables(x::AbstractVector, nmats::Int)
    d, r = divrem(length(x), nmats - 1)
    assert_eq(r, 0)
    return MultiMaterialVariables(reshape(x, d, nmats - 1))
end
function element_densities(x::PseudoDensities, densities::AbstractVector)
    return x.x * densities
end

function Base.sum(x::MultiMaterialVariables; dims)
    return sum(x.x; dims)
end

struct MaterialInterpolation{T,P}
    Es::Vector{T}
    penalty::P
end
function (f::MaterialInterpolation)(x::PseudoDensities)
    assert_eq(size(x.x, 2), length(f.Es))
    y = map(f.penalty, x.x) * f.Es
    return PseudoDensities(y)
end
function (f::MaterialInterpolation)(x::MultiMaterialVariables)
    assert_eq(size(x.x, 2), length(f.Es) - 1)
    y = map(f.penalty, tounit(x)) * f.Es
    return PseudoDensities(y)
end

function Utilities.setpenalty!(interp::MaterialInterpolation, p::Real)
    return Utilities.setpenalty!(interp.penalty, p)
end

tounit(x::MultiMaterialVariables) = PseudoDensities(tounit(x.x))

function tounit(x::AbstractVector)
    n = length(x) + 1
    T = eltype(x)
    stick = one(T)
    y = Vector{T}(undef, n)
    for i in 1:(n - 1)
        xi = x[i]
        z = logistic(xi - log(n - i))
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
