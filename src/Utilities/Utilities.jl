module Utilities

using ForwardDiff, Ferrite, IterativeSolvers, StaticArrays, LinearAlgebra

export AbstractPenalty,
    PowerPenalty,
    RationalPenalty,
    SinhPenalty,
    HeavisideProjection,
    SigmoidProjection,
    ProjectedPenalty,
    setpenalty,
    RaggedArray,
    @debug,
    compliance,
    sumdiag,
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

# Penalty definitions
include("penalties.jl")

end
