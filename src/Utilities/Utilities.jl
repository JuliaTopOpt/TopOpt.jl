module Utilities

using ForwardDiff, CUDAnative, CuArrays, ..GPUUtils, JuAFEM, IterativeSolvers

export  AbstractPenalty,
        PowerPenalty,
        RationalPenalty,
        TopOptTrace, 
        RaggedArray,
        @debug,
        compliance,
        meandiag,
        density, 
        find_black_and_white, 
        find_varind, 
        YoungsModulus,
        PoissonRatio,
        getpenalty,
        getprevpenalty,
        setpenalty!,
        getsolver,
        @params

function getpenalty end
function getprevpenalty end
function setpenalty! end
function getsolver end

# Utilities
include("utils.jl")

# Trace definition
include("traces.jl")

# Penalty definitions
include("penalties.jl")

end
