module Utilities

using ForwardDiff, Ferrite, IterativeSolvers, StaticArrays, LinearAlgebra

# Forward declarations extended by TopOptProblems
function getE end
function getν end

export AbstractPenalty,
    PowerPenaltyFun,
    RationalPenaltyFun,
    SinhPenaltyFun,
    HeavisideProjectionFun,
    SigmoidProjectionFun,
    ProjectedPenaltyFun,
    RaggedArray,
    @debug,
    compliance,
    sumdiag,
    meandiag,
    density,
    getpenalty,
    getprevpenalty,
    setpenalty!,
    getsolver,
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
"""
    getsolver(f)

Return the solver wrapped by `f` (a function or algorithm that stores its
solver in the `solver` field).
"""
getsolver(f) = f.solver

# Utilities
include("utils.jl")

# Penalty definitions
include("penalties.jl")

end
