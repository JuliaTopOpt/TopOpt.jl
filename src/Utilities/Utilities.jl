module Utilities

using ForwardDiff, Ferrite, IterativeSolvers, Requires

export  AbstractPenalty,
        PowerPenalty,
        RationalPenalty,
        HeavisideProjection,
        SigmoidProjection,
        ProjectedPenalty,
        setpenalty,
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
        @params,
        @forward_property

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
