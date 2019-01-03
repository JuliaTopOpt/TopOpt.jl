module Utilities

using ForwardDiff, CUDAnative, CuArrays

export  AbstractPenalty,
        PowerPenalty,
        RationalPenalty,
        TopOptTrace

# Utilities
include("utils.jl")

# Trace definition
include("traces.jl")

# Penalty definitions
include("penalties.jl")

end
