struct RaggedArray{TO,TV}
    offsets::TO
    values::TV
end

function RaggedArray(vv::Vector{Vector{T}}) where {T}
    offsets = [1; 1 .+ accumulate(+, collect(length(v) for v in vv))]
    values = Vector{T}(undef, offsets[end] - 1)
    for (i, v) in enumerate(vv)
        r = offsets[i]:(offsets[i + 1] - 1)
        values[r] .= v
    end
    return RaggedArray(offsets, values)
end

function Base.getindex(ra::RaggedArray, i)
    1 <= i < length(ra.offsets) || throw(BoundsError(ra, i))
    r = ra.offsets[i]:(ra.offsets[i + 1] - 1)
    1 <= r.start && r.stop <= length(ra.values) || throw(BoundsError(ra, (i, r)))
    return @view ra.values[r]
end
function Base.getindex(ra::RaggedArray, i, j)
    1 <= j < length(ra.offsets) || throw(BoundsError(ra, j))
    r = ra.offsets[j]:(ra.offsets[j + 1] - 1)
    1 <= i <= length(r) || throw(BoundsError(ra, (i, j)))
    return ra.values[r[i]]
end
function Base.setindex!(ra::RaggedArray, v, i, j)
    1 <= j < length(ra.offsets) || throw(BoundsError(ra, j))
    r = ra.offsets[j]:(ra.offsets[j + 1] - 1)
    1 <= i <= length(r) || throw(BoundsError(ra, (i, j)))
    return ra.values[r[i]] = v
end

function compliance(Ke, u, dofs)
    comp = zero(eltype(u))
    for i in eachindex(dofs)
        for j in eachindex(dofs)
            comp += u[dofs[i]] * Ke[i, j] * u[dofs[j]]
        end
    end
    return comp
end

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in axes(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end

"""
    density(var, xmin)

Map a design variable `var` ∈ [0, 1] to a physical density in `[xmin, 1]`:
`ρ = var * (1 - xmin) + xmin`. This is the interpolation step that gives void
(`xmin`) a small but nonzero stiffness to keep the system non-singular.
"""
density(var, xmin) = var * (1 - xmin) + xmin

macro debug(expr)
    return quote
        if DEBUG[]
            $(esc(expr))
        end
    end
end

@generated function _getproperty(c::T, ::Val{fallback}, ::Val{f}) where {T,fallback,f}
    f ∈ fieldnames(T) && return :(getfield(c, $(QuoteNode(f))))
    return :(getproperty(getfield(c, $(QuoteNode(fallback))), $(QuoteNode(f))))
end
@generated function _setproperty!(c::T, ::Val{fallback}, ::Val{f}, val) where {T,fallback,f}
    f ∈ fieldnames(T) && return :(setfield!(c, $(QuoteNode(f)), val))
    return :(setproperty!(getfield(c, $(QuoteNode(fallback))), $(QuoteNode(f)), val))
end
macro forward_property(T, field)
    quote
        function Base.getproperty(c::$(esc(T)), f::Symbol)
            return _getproperty(c, Val($(QuoteNode(field))), Val(f))
        end
        function Base.setproperty!(c::$(esc(T)), f::Symbol, val)
            return _setproperty!(c, Val($(QuoteNode(field))), Val(f), val)
        end
    end
end

for TM in (:(StaticMatrix{m,m,T}), :(Symmetric{T,<:StaticMatrix{m,m,T}}))
    @eval begin
        @generated function sumdiag(K::$TM) where {m,T}
            return reduce((ex1, ex2) -> :($ex1 + $ex2), [:(K[$j, $j]) for j in 1:m])
        end
    end
end
@doc """
sumdiag(K::Union{StaticMatrix, Symmetric{<:Any, <:StaticMatrix}})

Computes the sum of the diagonal of the static matrix `K`.
""" sumdiag
