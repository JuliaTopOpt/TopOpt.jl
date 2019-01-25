@params struct PolynomialFit{T}
    data::Tuple{Matrix{T}, Vector{T}}
    order::Ref{Int}
    coeffs::AbstractVector{T}
    ratio::T
end
PolynomialFit{T}(n, ratio) where T = PolynomialFit{T}((zeros(T, n+1, n+1), zeros(T, n+1)), Ref(-1), zeros(T, n+1), T(ratio))

function Base.convert(::Type{PolynomialFit}, n::Int)
    PolynomialFit{Float64}(n)
end
function Base.convert(::Type{PolynomialFit{T}}, n::Int) where T
    PolynomialFit{T}((zeros(T, n+1, n+1), zeros(T, n+1)), Ref(-1), 
        zeros(T, n+1), T(0.5))
end
Base.eltype(::PolynomialFit{T}) where T = T

# nth order derivatives of polynomial function f using f'''(x) notation
@inline Base.transpose(p::PolynomialFit) = Derivative(p)
@inline (poly::PolynomialFit)(x) = poly(x, 0)
@inline (poly::PolynomialFit)(x, n) = nthderivativeat(poly, n, x)
function nthderivativeat(poly::PolynomialFit{T}, n, x) where T
    o = zero(T)
    for j in (n+1):order(poly)+1
        p = n == 0 ? one(T) : prod(i->(j-i), 1:n)
        o += p*poly.coeffs[j]*x^(j-1-n)
    end
    return o
end

maxorder(p::PolynomialFit) = length(p.coeffs)-1
order(p::PolynomialFit) = p.order[]
increaseorder!(p::PolynomialFit) = p.order[]+= 1
reset!(p::PolynomialFit) = p.order[] = -1

previous(p::PolynomialFit, i) = i == 1 ? order(p) + 1 : i - 1
next(p::PolynomialFit, i) = i == order(p) + 1 ? 1 : i + 1

function newvalues!(p::PolynomialFit, X, V)
    for i in 1:length(X)
        newvalue!(p, X[i], V[i])
    end
end
function newvalue!(p::PolynomialFit{T}, x, v) where T
    if order(p) < maxorder(p)
        increaseorder!(p)
    end
    xs = [x^i for i in 0:maxorder(p)]
    p.data[1] .= (1 .- p.ratio) .* p.data[1] .+ p.ratio .* (xs * xs')
    p.data[2] .= (1 .- p.ratio) .* p.data[2] .+ p.ratio .* xs .* v
end

function newderivatives!(p::PolynomialFit, X, V)
    for i in 1:length(X)
        newderivative!(p, X[i], V[i])
    end
end
function newderivative!(p::PolynomialFit{T}, x, v) where T
    if order(p) < maxorder(p)
        increaseorder!(p)
    end
    xs = [(j-1)x^(j-2) for j in 1:maxorder(p)+1]
    p.data[1] .= (1 .- p.ratio) .* p.data[1] .+ p.ratio .* (xs * xs')
    p.data[2] .= (1 .- p.ratio) .* p.data[2] .+ p.ratio .* xs .* v
end

function solve!(p::PolynomialFit)
    n = order(p)+1
    p.coeffs[1:n] .= pinv(p.data[1][1:n, 1:n]) * p.data[2][1:n]
    @view(p.coeffs[1:n])
end

function roots(poly::Union{TP, Derivative{TP}}, min=T(0), max=T(15)) where {T, TP<:PolynomialFit{T}}
    root_ranges = IntervalRootFinding.roots(x->poly(x), min..max)
    return _filter(root_ranges)
end

Base.:/(p1::PolynomialFit, p2::PolynomialFit) = PolynomialQuotient(p1, p2)

Base.:(==)(p::PolynomialFit, n) = PolynomialEquality(p, n)

function roots(poly::PolynomialEquality{TP}, min=eltype(poly)(0), max=eltype(poly)(15)) where {TP<:Union{PolynomialFit, Derivative{<:PolynomialFit}}}
    root_ranges = IntervalRootFinding.roots(x->(poly.LHS(x) - poly.RHS), min..max)
    return _filter(root_ranges)
end

# TO-DO: derivative of a quotient using the quotient rule
