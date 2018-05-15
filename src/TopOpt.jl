module TopOpt

using Reexport
@reexport using TopOptProblems
using JuAFEM
using TimerOutputs
using ForwardDiff
using MMA
@reexport using Optim

norm(a) = sqrt(dot(a,a))

include("utils.jl")
include("types.jl")
include("assemble.jl")
include("matrix_free_apply_bcs.jl")
include("simulate.jl")
include("solvers_api.jl")
include("math_optimizers.jl")
include("simp.jl")
include("continuation_schemes.jl")
include("continuation_simp.jl")
include("writevtk.jl")

export simulate, TopOptTrace, VolConstr, DirectDisplacementSolver, PCGDisplacementSolver, StaticMatrixFreeDisplacementSolver, CheqFilter, ComplianceObj, Displacement, CG, Direct, Assembly, MatrixFree, FEASolver, MMAOptimizer, SIMP, ContinuationSIMP, PowerContinuation, ExponentialContinuation, LogarithmicContinuation, CubicSplineContinuation, SigmoidContinuation, save_mesh

end