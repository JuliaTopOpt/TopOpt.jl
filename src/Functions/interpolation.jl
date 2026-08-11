assert_eq(x1, x2) = @assert x1 == x2
function ChainRulesCore.rrule(::typeof(assert_eq), x1, x2)
    return assert_eq(x1, x2), _ -> (NoTangent(), NoTangent(), NoTangent())
end

"""
    MultiMaterialVariables(y, nmats)

Wraps the raw per-cell, per-material decision variables `y` (length
`ncells * (nmats - 1)`) for use with `MaterialInterpolation`.
"""
struct MultiMaterialVariablesFun{M<:AbstractMatrix}
    x::M
end
function MultiMaterialVariablesFun(x::AbstractVector, nmats::Int)
    d, r = divrem(length(x), nmats - 1)
    assert_eq(r, 0)
    return MultiMaterialVariablesFun(reshape(x, d, nmats - 1))
end
"""
    element_densities(mv::MultiMaterialVariablesFun)

Extract the per-element density vector from multi-material variables.
"""
function element_densities(x::PseudoDensities, densities::AbstractVector)
    return x.x * densities
end

function Base.sum(x::MultiMaterialVariablesFun; dims)
    return sum(x.x; dims)
end

"""
    MaterialInterpolation(values, penalty)

Maps a softmax over per-material decision variables to a physical material
property (e.g. Young's modulus or density). `values` is a vector of material
property values (length `nmats`, including void as the first entry). `penalty`
is applied to the softmax output.
"""
struct MaterialInterpolationFun{T,P}
    Es::Vector{T}
    penalty::P
end
function (f::MaterialInterpolationFun)(x::PseudoDensities)
    assert_eq(size(x.x, 2), length(f.Es))
    y = map(f.penalty, x.x) * f.Es
    return PseudoDensities(y)
end
function (f::MaterialInterpolationFun)(x::MultiMaterialVariablesFun)
    assert_eq(size(x.x, 2), length(f.Es) - 1)
    return f(tounit(x))
end

function Utilities.setpenalty!(interp::MaterialInterpolationFun, p::Real)
    return Utilities.setpenalty!(interp.penalty, p)
end

"""
    tounit(mv::MultiMaterialVariablesFun)

Convert `MultiMaterialVariables` to unit-sum densities via softmax, so the
per-cell material fractions sum to 1.
"""
tounit(x::MultiMaterialVariablesFun) = PseudoDensities(tounit(x.x))

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
    return tounit(x),
    Δ -> (NoTangent(), ForwardDiff.jacobian(tounit, x)' * ChainRulesCore.unthunk(Δ))
end
function ChainRulesCore.rrule(::typeof(tounit), x::Matrix)
    pb = (x, Δ) -> (ForwardDiff.jacobian(tounit, x)' * ChainRulesCore.unthunk(Δ))'
    return tounit(x),
    Δ -> (NoTangent(), mapreduce(pb, vcat, eachrow(x), eachrow(ChainRulesCore.unthunk(Δ))))
end
