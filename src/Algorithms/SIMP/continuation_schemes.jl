abstract type AbstractContinuation <: Function end

struct FixedContinuation{T} <: AbstractContinuation
    param::T
    length::Int
end
function Base.iterate(s::FixedContinuation, x=1)
    (x > s.length) && return nothing
    s.param, x+1
end
Base.length(s::FixedContinuation) = s.length
(s::FixedContinuation{T})(x...) where T = s.param

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
function SigmoidContinuation{T}(; c::T=T(0.1), start::T=T(1), finish::T=T(5), steps::Int=30, min::T=-Inf) where T
    a = 1 - T(finish-start)/finish/(ℯ^(-c) - ℯ^(-steps*c)) * ℯ^(-c)
    b = T(finish-start)/finish/(ℯ^(-c) - ℯ^(-steps*c))
    SigmoidContinuation{T}(a, b, c, steps, min)
end
function Base.iterate(s::SigmoidContinuation, x=1)
    (x > s.length) && return nothing
    max(1/(s.a + s.b*ℯ^(-s.c*x)), s.min), x+1
end
Base.length(s::SigmoidContinuation) = s.length
(s::SigmoidContinuation{T})(x) where T = max(1/(s.a + s.b*ℯ^(-s.c*T(x))), s.min)

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
function CubicSplineContinuation{T}(;b::T=T(0.5), start::T=T(1), finish::T=T(5), steps::Int=30, min::T=-Inf) where T
    a = (finish-start)/(steps^3*(1-b)^3 - (1-b*steps)^3)
    c = start - a*(1-b*steps)^3
    CubicSplineContinuation{T}(a,b,c,steps,min)
end

function Base.iterate(s::CubicSplineContinuation, x=1)
    x > s.length && return nothing
    max(s.a*(x-s.b*s.length)^3 + s.c, s.min), x+1
end
Base.length(s::CubicSplineContinuation) = s.length
(s::CubicSplineContinuation{T})(x) where T = max(s.a*(T(x)-s.b*s.length)^3 + s.c, s.min)

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
function PowerContinuation{T}(;b::T=T(2), start::T=T(1), finish::T=T(5), steps::Int=30, min::T=-Inf) where T
    a = (finish - start) / max(T(1), steps^b - 1)
    c = start - a
    PowerContinuation{T}(a, b, c, steps, min)
end
function Base.iterate(s::PowerContinuation, x=1)
    x > s.length && return nothing
    max(s.a*x^s.b + s.c, s.min), x+1
end
Base.length(s::PowerContinuation) = s.length
(s::PowerContinuation{T})(x) where T = max(s.a*T(x)^s.b + s.c, s.min)

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
function ExponentialContinuation{T}(;b::T=T(0.1), start::T=T(1), finish::T=T(5), steps::Int=30, min::T=-Inf) where T
    a = (finish - start) / max(T(1), ℯ^(b*steps) - ℯ^(b))
    c = start - a*ℯ^(b)
    ExponentialContinuation{T}(a,b,c,steps,min)
end
function Base.iterate(s::ExponentialContinuation, x=1)
    x > s.length && return nothing
    max(s.a*ℯ^(s.b*x) + s.c, s.min), x+1
end
Base.length(s::ExponentialContinuation) = s.length
(s::ExponentialContinuation{T})(x) where T = max(s.a*ℯ^(s.b*T(x)) + s.c, s.min)

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
function LogarithmicContinuation{T}(;b::T=T(1), start::T=T(1), finish::T=T(5), steps::Int=30, min::T=-Inf) where T
    a = (finish - start) / max(T(1), log(b*steps) - log(b))
    c = start - a*log(b)
    LogarithmicContinuation{T}(a,b,c,steps,min)
end
function Base.iterate(s::LogarithmicContinuation, x=1)
    x > s.length && return nothing
    max(s.a*log(s.b*x) + s.c, s.min), x+1
end
Base.length(s::LogarithmicContinuation) = s.length
(s::LogarithmicContinuation{T})(x) where T = max(s.a*log(s.b*T(x)) + s.c, s.min)
