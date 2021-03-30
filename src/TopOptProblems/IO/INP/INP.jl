module INP

export  InpStiffness

using ...TopOptProblems: Metadata, StiffnessTopOptProblem
using Ferrite
using ....TopOpt.Utilities: find_black_and_white, find_varind
import ...TopOptProblems: nnodespercell, getE, getν, getgeomorder, getdensity, getpressuredict, getcloaddict, getfacesets

include(joinpath("Parser", "Parser.jl"))
using .Parser

include("inpstiffness.jl")

end
