abstract type AbstractContinuation <: Function end

struct RationalContinuation{T}
    pmax::T
    kmax::Int
    b::T
end
function Base.getproperty(c::RationalContinuation, f::Symbol)
    f === :length && return length(c)
    return getfield(c, f)
end
function RationalContinuation(xmin, steps::Int)
    pmax = (1 - xmin) / xmin
    kmax = steps
    b = 1 / steps * (1 - sqrt(xmin)) / (1 + sqrt(xmin))
    return RationalContinuation(pmax, kmax, b)
end
function Base.iterate(s::RationalContinuation, k = 1)
    (k > s.kmax + 1) && return nothing
    b = s.b
    return s(k), k + 1
end
Base.length(s::RationalContinuation) = s.kmax + 1
function (s::RationalContinuation{T})(k) where {T}
    k = clamp(k, 1, s.kmax + 1)
    b = s.b
    return 4 * b * (k - 1) / (1 - b * (k - 1))^2
end

struct FixedContinuation{T} <: AbstractContinuation
    param::T
    length::Int
end
function Base.iterate(s::FixedContinuation, x = 1)
    (x > s.length) && return nothing
    s.param, x + 1
end
Base.length(s::FixedContinuation) = s.length
(s::FixedContinuation{T})(x...) where {T} = s.param

"""
p(x) = 1/(a + b*ℯ^(-c*x))
"""
struct SigmoidContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
    min::T
end
SigmoidContinuation(; kwargs...) = SigmoidContinuation{Float64}(; kwargs...)
function SigmoidContinuation{T}(;
    c::T = T(0.1),
    start::T = T(1),
    finish::T = T(5),
    steps::Int = 30,
    min::T = -Inf,
) where {T}
    a = 1 - T(finish - start) / finish / (ℯ^(-c) - ℯ^(-steps * c)) * ℯ^(-c)
    b = T(finish - start) / finish / (ℯ^(-c) - ℯ^(-steps * c))
    SigmoidContinuation(a, b, c, steps, min)
end
function Base.iterate(s::SigmoidContinuation, x = 1)
    (x > s.length) && return nothing
    max(1 / (s.a + s.b * ℯ^(-s.c * x)), s.min), x + 1
end
Base.length(s::SigmoidContinuation) = s.length
(s::SigmoidContinuation{T})(x) where {T} = max(1 / (s.a + s.b * ℯ^(-s.c * T(x))), s.min)

"""
p(x) = a(x-b*steps)^3 + c
"""
struct CubicSplineContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
    min::T
end
CubicSplineContinuation(; kwargs...) = CubicSplineContinuation{Float64}(; kwargs...)
function CubicSplineContinuation{T}(;
    b::T = T(0.5),
    start::T = T(1),
    finish::T = T(5),
    steps::Int = 30,
    min::T = -Inf,
) where {T}
    a = (finish - start) / (steps^3 * (1 - b)^3 - (1 - b * steps)^3)
    c = start - a * (1 - b * steps)^3
    CubicSplineContinuation(a, b, c, steps, min)
end

function Base.iterate(s::CubicSplineContinuation, x = 1)
    x > s.length && return nothing
    max(s.a * (x - s.b * s.length)^3 + s.c, s.min), x + 1
end
Base.length(s::CubicSplineContinuation) = s.length
(s::CubicSplineContinuation{T})(x) where {T} =
    max(s.a * (T(x) - s.b * s.length)^3 + s.c, s.min)

"""
p(x) = a*x^b + c
"""
struct PowerContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
    min::T
end
PowerContinuation(; kwargs...) = PowerContinuation{Float64}(; kwargs...)
function PowerContinuation{T}(;
    b = T(2),
    start = T(1),
    finish = T(5),
    steps::Int = 30,
    min = -T(Inf),
) where {T}
    a = (finish - start) / max(T(1), steps^b - 1)
    c = start - a
    PowerContinuation(a, b, c, steps, min)
end
function Base.iterate(s::PowerContinuation, x = 1)
    x > s.length && return nothing
    max(s.a * x^s.b + s.c, s.min), x + 1
end
Base.length(s::PowerContinuation) = s.length
(s::PowerContinuation{T})(x) where {T} = max(s.a * T(x)^s.b + s.c, s.min)

"""
p(x) = a*ℯ^(b*x) + c
"""
struct ExponentialContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
    min::T
end
ExponentialContinuation(; kwargs...) = ExponentialContinuation{Float64}(; kwargs...)
function ExponentialContinuation{T}(;
    b::T = T(0.1),
    start::T = T(1),
    finish::T = T(5),
    steps::Int = 30,
    min::T = -Inf,
) where {T}
    a = (finish - start) / max(T(1), ℯ^(b * steps) - ℯ^(b))
    c = start - a * ℯ^(b)
    ExponentialContinuation(a, b, c, steps, min)
end
function Base.iterate(s::ExponentialContinuation, x = 1)
    x > s.length && return nothing
    max(s.a * ℯ^(s.b * x) + s.c, s.min), x + 1
end
Base.length(s::ExponentialContinuation) = s.length
(s::ExponentialContinuation{T})(x) where {T} = max(s.a * ℯ^(s.b * T(x)) + s.c, s.min)

"""
p(x) = a*log(b*x) + c
"""
struct LogarithmicContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
    min::T
end
LogarithmicContinuation(; kwargs...) = LogarithmicContinuation{Float64}(; kwargs...)
function LogarithmicContinuation{T}(;
    b::T = T(1),
    start::T = T(1),
    finish::T = T(5),
    steps::Int = 30,
    min::T = -Inf,
) where {T}
    a = (finish - start) / max(T(1), log(b * steps) - log(b))
    c = start - a * log(b)
    LogarithmicContinuation(a, b, c, steps, min)
end
function Base.iterate(s::LogarithmicContinuation, x = 1)
    x > s.length && return nothing
    max(s.a * log(s.b * x) + s.c, s.min), x + 1
end
Base.length(s::LogarithmicContinuation) = s.length
(s::LogarithmicContinuation{T})(x) where {T} = max(s.a * log(s.b * T(x)) + s.c, s.min)

Continuation(::PowerPenalty; kwargs...) = Continuation(PowerPenalty; kwargs...)
Continuation(::RationalPenalty; kwargs...) = Continuation(RationalPenalty; kwargs...)
function Continuation(::Type{<:PowerPenalty}; steps, pmax = 5.0, kwargs...)
    return PowerContinuation{Float64}(
        b = 1.0,
        start = 1.0,
        steps = steps + 1,
        finish = pmax,
    )
end
function Continuation(::Type{<:RationalPenalty}; steps, xmin = 0.001, kwargs...)
    return RationalContinuation(xmin, steps)
end
