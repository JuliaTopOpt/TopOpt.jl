module Visualization

using ..TopOptProblems: AbstractTopOptProblem, StiffnessTopOptProblem, QuadraticHexahedron
using Ferrite, VTKDataTypes, Requires

function visualize end

include("mesh_types.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" @eval begin
            include("../../../ext/MakieExt.jl")
            export visualize
        end
    end
end

end
