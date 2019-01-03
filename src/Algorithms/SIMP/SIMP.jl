abstract type AbstractSIMP <: TopOptAlgorithm end

# MMA wrapper
include("math_optimizers.jl")

# Basic SIMP
include("basic_simp.jl")

## Continuation SIMP
include("continuation_schemes.jl")
include("continuation_simp.jl")

## Adaptive SIMP
include("polynomials.jl")
include("polynomials2.jl")
include("adaptive_simp.jl")
