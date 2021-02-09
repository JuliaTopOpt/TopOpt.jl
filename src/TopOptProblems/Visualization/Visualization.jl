module Visualization

using ..TopOptProblems: AbstractTopOptProblem, StiffnessTopOptProblem
using JuAFEM, VTKDataTypes

# export ...
include("mesh_types.jl")
include("makie.jl")

end