@params struct PartialPolynomialFit{T}
    data::Tuple{Matrix{T}, Vector{T}}
    order::Ref{Int}
    coeffs::AbstractVector{T}
    oldest::Ref{Int}
end
function Base.convert(::Type{PartialPolynomialFit}, n::Int)
    PartialPolynomialFit{Float64}(n)
end
function Base.convert(::Type{PartialPolynomialFit{T}}, n::Int) where T
    PartialPolynomialFit{T}((zeros(T, n+1, n+1), zeros(T, n+1)), Ref(-1), 
        zeros(T, n+1), Ref(1))
end
Base.eltype(::PartialPolynomialFit{T}) where T = T

@params struct Derivative
    f
end
# nth order derivatives of polynomial function f using f'''(x) notation
@inline Base.transpose(p::Union{PartialPolynomialFit, Derivative}) = Derivative(p)
@inline (poly::PartialPolynomialFit)(x) = poly(x, 0)
@inline (poly::PartialPolynomialFit)(x, n) = nthderivativeat(poly, n, x)
@inline (der::Derivative)(x) = der.f(x, 1)
@inline (der::Derivative)(x, n) = der.f(x, n+1)
function nthderivativeat(poly::PartialPolynomialFit{T}, n, x) where T
    o = zero(T)
    for j in (n+1):order(poly)+1
        p = n == 0 ? one(T) : prod(i->(j-i), 1:n)
        o += p*poly.coeffs[j]*x^(j-1-n)
    end
    return o
end
Base.eltype(d::Derivative) = eltype(d.f)

maxorder(p::PartialPolynomialFit) = length(p.coeffs)-1
order(p::PartialPolynomialFit) = p.order[]
increaseorder!(p::PartialPolynomialFit) = p.order[]+= 1
reset!(p::PartialPolynomialFit) = p.order[] = -1
oldest(p::PartialPolynomialFit) = p.oldest[]
newest(p::PartialPolynomialFit) = p.oldest[] == 1 ? order(p) + 1 : p.oldest[] - 1

previous(p::PartialPolynomialFit, i) = i == 1 ? order(p) + 1 : i - 1
next(p::PartialPolynomialFit, i) = i == order(p) + 1 ? 1 : i + 1

function newvalues!(p::PartialPolynomialFit, X, V)
    for i in 1:length(X)
        newvalue!(p, X[i], V[i])
    end
    oldest(p)
end
function newvalue!(p::PartialPolynomialFit, x, v)
    n = incrementoldest!(p)
    if order(p) < maxorder(p)
        increaseorder!(p)
    end
    newvalue!(p, x, v, n)
end
function newvalue!(p::PartialPolynomialFit, x, v, i)
    for j in 1:maxorder(p)+1
        p.data[1][i,j] = x^(j-1)
    end
    p.data[2][i] = v
    oldest(p)
end

function newderivatives!(p::PartialPolynomialFit, X, V)
    for i in 1:length(X)
        newderivative!(p, X[i], V[i])
    end
    oldest(p)
end
function newderivative!(p::PartialPolynomialFit, x, v)
    n = incrementoldest!(p)
    if order(p) < maxorder(p)
        increaseorder!(p)
    end
    newderivative!(p, x, v, n)
end
function newderivative!(p::PartialPolynomialFit, x, v, i)
    for j in 1:maxorder(p)+1
        p.data[1][i,j] = (j-1)x^(j-2)
    end
    p.data[2][i] = v
    oldest(p)
end

function incrementoldest!(p::PartialPolynomialFit)
    o = p.oldest[]
    if o == maxorder(p)+1
        p.oldest[] = 1
    else
        p.oldest[] = o + 1
    end
    o
end

function solve!(p::PartialPolynomialFit)
    n = order(p)+1
    p.coeffs[1:n] .= p.data[1][1:n, 1:n] \ p.data[2][1:n]
    @view(p.coeffs[1:n])
end

function _filter(root_ranges)
    unique_root_ranges = filter!(x->(x.status == :unique), root_ranges)
    return [(r.interval.lo + r.interval.lo)/2 for r in unique_root_ranges]
end

function _filter(optimiser_ranges::Tuple)
    return [(r.lo + r.lo)/2 for r in optimiser_ranges[2]]
end

function roots(poly::Union{TP, Derivative{TP}}, min=T(0), max=T(15)) where {T, TP<:PartialPolynomialFit{T}}
    root_ranges = IntervalRootFinding.roots(x->poly(x), min..max)
    return _filter(root_ranges)
end

@params struct PolynomialQuotient
    num
    denom
end
Base.:/(p1::Union{PartialPolynomialFit, Derivative}, p2::Union{PartialPolynomialFit, Derivative}) = PolynomialQuotient(p1, p2)
Base.eltype(p::PolynomialQuotient) = promote_type(eltype(p.num), eltype(p.denom))

@params struct PolynomialEquality
    LHS
    RHS
end
Base.:(==)(p::Union{PolynomialQuotient, PartialPolynomialFit, Derivative}, n) = PolynomialEquality(p, n)
Base.eltype(p::PolynomialEquality) = promote_type(eltype(p.LHS), typeof(p.RHS))

function roots(poly::PolynomialEquality{TP}, min=eltype(poly)(0), max=eltype(poly)(15)) where {TP<:Union{PartialPolynomialFit, Derivative{<:PartialPolynomialFit}}}
    root_ranges = IntervalRootFinding.roots(x->(poly.LHS(x) - poly.RHS), min..max)
    return _filter(root_ranges)
end

function roots(poly::PolynomialEquality{<:PolynomialQuotient}, min=eltype(poly)(0), max=eltype(poly)(15))
    root_ranges = IntervalRootFinding.roots(x->(poly.LHS.num(x) - poly.RHS * poly.LHS.denom(x)), min..max)
    return _filter(root_ranges)
end

# TO-DO: derivative of a quotient using the quotient rule
