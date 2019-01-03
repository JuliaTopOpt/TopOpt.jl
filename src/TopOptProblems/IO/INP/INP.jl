module INP

export  InpStiffness

using ...TopOptProblems: Metadata, StiffnessTopOptProblem
using JuAFEM

include(joinpath("InpParser", "InpParser.jl"))
include("inpstiffness.jl")

end
