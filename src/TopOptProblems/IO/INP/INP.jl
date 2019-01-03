module INP

export  InpStiffness

using ...TopOptProblems: Metadata, StiffnessTopOptProblem
using JuAFEM

include(joinpath("Parser", "Parser.jl"))
using .Parser

include("inpstiffness.jl")

end
