module Visualization

using ..TopOptProblems: AbstractTopOptProblem, StiffnessTopOptProblem, QuadraticHexahedron
using Ferrite, VTKDataTypes, Requires

include("mesh_types.jl")

function __init__()
    @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" @eval begin
        include("makie.jl")
        export visualize
    end
end

end
