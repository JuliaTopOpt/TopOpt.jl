module InputOutput

export InpStiffness, save_mesh

include("mesh_types.jl")

include(joinpath("INP", "INP.jl"))
using .INP

include("VTK.jl")
using .VTK

end
