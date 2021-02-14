abstract type AbstractSIMP <: TopOptAlgorithm end

# MMA wrapper
include("mma_optimizer.jl")

# Basic SIMP
include("basic_simp.jl")

## Continuation SIMP
include("continuation_schemes.jl")
include("continuation_simp.jl")
