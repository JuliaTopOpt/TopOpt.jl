module TrussIO

import FileIO
import FileIO: skipmagic, add_format

using ..TrussTopOptProblems: TrussFEACrossSec, TrussFEAMaterial

using StaticArrays

include("parse_json.jl")
include("parse_geo.jl")

# function load(fn::File{format}) where {format}
#     open(fn) do s
#         skipmagic(s)
#         load(s)
#     end
# end

export load_truss_json, load_truss_geo

end
