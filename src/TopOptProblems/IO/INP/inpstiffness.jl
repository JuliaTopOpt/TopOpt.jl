"""
Stiffness problem imported from a .inp file.
"""
struct InpStiffness{dim, N, TF, M, TI, TBool, GO, TInds <: AbstractVector{TI}, TMeta<:Metadata} <: StiffnessTopOptProblem{dim, TF}
    inp_content::InpContent{dim, TF, N, TI}
    geom_order::Type{Val{GO}}
    ch::ConstraintHandler{DofHandler{dim, N, TF, M}, TF}
    black::TBool
    white::TBool
    varind::TInds
    metadata::TMeta
end

"""
Imports stiffness problem from a .inp file.
"""
function InpStiffness(filepath_with_ext::AbstractString)
    problem = Parser.extract_inp(filepath_with_ext)
    return InpStiffness(problem)
end
function InpStiffness(problem::Parser.InpContent)
    ch = Parser.inp_to_juafem(problem)
    black, white = find_black_and_white(ch.dh)
    varind = find_varind(black, white)
    metadata = Metadata(ch.dh)
    geom_order = JuAFEM.getorder(ch.dh.field_interpolations[1])
    return InpStiffness(problem, Val{geom_order}, ch, black, white, varind, metadata)
end

getE(p::InpStiffness) = p.inp_content.E
getν(p::InpStiffness) = p..inp_content.ν
nnodespercell(::InpStiffness{dim, N}) where {dim, N} = N
getgeomorder(p::InpStiffness{dim, N, TF, M, TI, GO}) where {dim, N, TF, M, TI, GO} = GO
getdensity(p::InpStiffness) = p.inp_content.density
getpressuredict(p::InpStiffness) = p.inp_content.dloads
getcloaddict(p::InpStiffness) = p.inp_content.cloads
getfacesets(p::InpStiffness) = p.inp_content.facesets
