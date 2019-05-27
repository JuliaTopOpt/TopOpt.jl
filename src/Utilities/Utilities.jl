module Utilities

using ForwardDiff, JuAFEM, IterativeSolvers, Requires

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
getsolver(f) = f.solver

# Utilities
include("utils.jl")

# Trace definition
include("traces.jl")

# Penalty definitions
include("penalties.jl")

end
