module Visualization

using ..TopOptProblems: AbstractTopOptProblem, StiffnessTopOptProblem
using Ferrite, VTKDataTypes, Requires

# export ...
include("mesh_types.jl")

function __init__()
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" @eval include("makie.jl")
end

end