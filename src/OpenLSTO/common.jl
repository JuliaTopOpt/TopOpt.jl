# Common data types (a port of `M2DO_LSM/include/common.h`).

struct Coord
    x::Float64
    y::Float64
end

mutable struct BoundaryPoint
    coord::Coord
    normal::Coord
    length::Float64
    velocity::Float64
    negativeLimit::Float64
    positiveLimit::Float64
    isDomain::Bool
    isFixed::Bool
    segments::Vector{Int}
    neighbours::Vector{Int}
    sensitivities::Vector{Float64}
end

mutable struct BoundarySegment
    start::Int
    stop::Int
    element::Int
    length::Float64
    weight::Float64
end
