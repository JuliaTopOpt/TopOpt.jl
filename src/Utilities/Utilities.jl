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
    YoungsModulus,
    PoissonRatio,
    getpenalty,
    getprevpenalty,
    setpenalty!,
    getsolver,
    @params,
    @forward_property

"""
    getpenalty(solver)

Return the current penalty object of `solver` (or of a function wrapping a
solver).
"""
function getpenalty end
"""
    getprevpenalty(solver)

Return the previous penalty object (before the last `setpenalty!` call).
"""
function getprevpenalty end
"""
    setpenalty!(solver, p)

Update the penalty of `solver` to `p` (a number or an `AbstractPenalty`).
Stashes the old penalty in `getprevpenalty`.
"""
function setpenalty! end
getsolver(f) = f.solver

# Utilities
include("utils.jl")

# Penalty definitions
include("penalties.jl")

end
