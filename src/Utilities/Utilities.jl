module Utilities

using ForwardDiff, CUDAnative, CuArrays, ..GPUUtils, JuAFEM

export  AbstractPenalty,
        PowerPenalty,
        RationalPenalty,
        TopOptTrace, 
        RaggedArray,
        @debug,
        compliance,
        meandiag,
        density, find_black_and_white, 
        find_varind, 
        YoungsModulus,
        PoissonRatio

# Utilities
include("utils.jl")

# Trace definition
include("traces.jl")

# Penalty definitions
include("penalties.jl")

end
