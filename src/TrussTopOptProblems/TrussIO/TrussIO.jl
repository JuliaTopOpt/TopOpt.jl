module TrussIO

using ..TrussTopOptProblems: TrussFEACrossSec, TrussFEAMaterial

using StaticArrays
export  parse_truss_json,
        parse_support_load_json

include("parse_json.jl")

end

