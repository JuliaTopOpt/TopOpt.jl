using Missings

abstract type AbstractContinuation end

"""
p(x) = 1/(a + b*e^(-c*x))
"""
struct SigmoidContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
end
function SigmoidContinuation{T}(;c::T=T(0.1), start::T=T(1), finish::T=T(5), steps::Int=30) where T
    a = 1 - T(finish-start)/finish/(e^(-c) - e^(-steps*c)) * e^(-c)
    b = T(finish-start)/finish/(e^(-c) - e^(-steps*c))
    SigmoidContinuation{T}(a,b,c,steps)
end
Base.start(::SigmoidContinuation) = 1
Base.next(s::SigmoidContinuation, x) = 1/(s.a + s.b*e^(-s.c*x)), x+1
Base.done(s::SigmoidContinuation, x) = x > s.length
Base.length(s::SigmoidContinuation) = s.length
(s::SigmoidContinuation{T})(x) where T = 1/(s.a + s.b*e^(-s.c*T(x)))

"""
p(x) = a(x-b*steps)^3 + c
"""
struct CubicSplineContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
end
function CubicSplineContinuation{T}(;b::T=T(0.5), start::T=T(1), finish::T=T(5), steps::Int=30) where T
    a = (finish-start)/(steps^3*(1-b)^3 - (1-b*steps)^3)
    c = start - a*(1-b*steps)^3
    CubicSplineContinuation{T}(a,b,c,steps)
end
Base.start(::CubicSplineContinuation) = 1
Base.next(s::CubicSplineContinuation, x) = s.a*(x-s.b*s.length)^3 + s.c, x+1
Base.done(s::CubicSplineContinuation, x) = x > s.length
Base.length(s::CubicSplineContinuation) = s.length
(s::CubicSplineContinuation{T})(x) where T = s.a*(T(x)-s.b*s.length)^3 + s.c

"""
p(x) = a*x^b + c
"""
struct PowerContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
end
function PowerContinuation{T}(;b::T=T(2), start::T=T(1), finish::T=T(5), steps::Int=30) where T
    a = (finish - start) / max(T(1), steps^b - 1)
    c = start - a
    PowerContinuation{T}(a,b,c,steps)
end
Base.start(::PowerContinuation) = 1
Base.next(s::PowerContinuation, x) = s.a*x^s.b + s.c, x+1
Base.done(s::PowerContinuation, x) = x > s.length
Base.length(s::PowerContinuation) = s.length
(s::PowerContinuation{T})(x) where T = s.a*T(x)^s.b + s.c

"""
p(x) = a*e^(b*x) + c
"""
struct ExponentialContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
end
function ExponentialContinuation{T}(;b::T=T(0.1), start::T=T(1), finish::T=T(5), steps::Int=30) where T
    a = (finish - start) / max(T(1), e^(b*steps) - e^(b))
    c = start - a*e^(b)
    ExponentialContinuation{T}(a,b,c,steps)
end
Base.start(::ExponentialContinuation) = 1
Base.next(s::ExponentialContinuation, x) = s.a*e^(s.b*x) + s.c, x+1
Base.done(s::ExponentialContinuation, x) = x > s.length
Base.length(s::ExponentialContinuation) = s.length
(s::ExponentialContinuation{T})(x) where T = s.a*e^(s.b*T(x)) + s.c

"""
p(x) = a*log(b*x) + c
"""
struct LogarithmicContinuation{T} <: AbstractContinuation
    a::T
    b::T
    c::T
    length::Int
end
function LogarithmicContinuation{T}(;b::T=T(1), start::T=T(1), finish::T=T(5), steps::Int=30) where T
    a = (finish - start) / max(T(1), log(b*steps) - log(b))
    c = start - a*log(b)
    LogarithmicContinuation{T}(a,b,c,steps)
end
Base.start(::LogarithmicContinuation) = 1
Base.next(s::LogarithmicContinuation, x) = s.a*log(s.b*x) + s.c, x+1
Base.done(s::LogarithmicContinuation, x) = x > s.length
Base.length(s::LogarithmicContinuation) = s.length
(s::LogarithmicContinuation{T})(x) where T = s.a*log(s.b*T(x)) + s.c
