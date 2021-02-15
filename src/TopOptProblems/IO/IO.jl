module InputOutput

export  InpStiffness,
        save_mesh

include(joinpath("INP", "INP.jl"))
using .INP

include("VTK.jl")
using .VTK

end
