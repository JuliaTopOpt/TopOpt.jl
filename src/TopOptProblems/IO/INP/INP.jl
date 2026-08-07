module INP

export InpStiffness

using ...TopOptProblems: Metadata, StiffnessTopOptProblem, _base_interpolation
using Ferrite
import ...TopOptProblems:
    nnodespercell,
    getE,
    getν,
    getgeomorder,
    getdensity,
    getpressuredict,
    getcloaddict,
    getfacesets

include(joinpath("Parser", "Parser.jl"))
using .Parser

include("inpstiffness.jl")

end
