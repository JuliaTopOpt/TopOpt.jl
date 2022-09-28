struct MultiMaterialPseudoDensities{M <: AbstractMatrix}
    x::M
end

function element_densities(x::MultiMaterialPseudoDensities, densities::AbstractVector)
    return x.x * densities
end

function sum(x::MultiMaterialPseudoDensities; dims)
    return sum(x.x; dims)
end

struct MaterialInterpolation{T, P}
    E0::T
    ΔEs::Vector{T}
    penalty::P
end
function MaterialInterpolation(Es::Vector, penalty::AbstractPenalty)
    E0 = first(Es)
    ΔEs = @view(Es[2:end]) .- E0
    return MaterialInterpolation(E0, ΔEs, penalty)
end
(f::MaterialInterpolation)(x::PseudoDensities) = f(x.x)
(f::MaterialInterpolation)(x::MultiMaterialPseudoDensities) = f(x.x)
function (f::MaterialInterpolation)(x::AbstractVector)
    d, r = divrem(length(x), length(f.ΔEs))
    @assert r == 0
    return f(reshape(x, d, length(f.ΔEs)))
end
function (f::MaterialInterpolation)(x::AbstractMatrix)
    @assert size(x, 2) == length(f.ΔEs)
    y = map(f.penalty, x) * f.ΔEs .+ f.E0
    return PseudoDensities(y)
end
