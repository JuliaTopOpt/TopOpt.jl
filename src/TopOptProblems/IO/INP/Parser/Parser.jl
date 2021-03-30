module Parser

using Ferrite, SparseArrays
using .....TopOpt: find_black_and_white

export extract_inp, InpContent

"""
```
struct InpContent{dim, TF, N, TI}
    node_coords::Vector{NTuple{dim,TF}}
    celltype::String
    cells::Vector{NTuple{N,TI}}
    nodesets::Dict{String,Vector{TI}}
    cellsets::Dict{String,Vector{TI}}
    E::TF
    ν::TF
    density::TF
    nodedbcs::Dict{String, Vector{Tuple{TI,TF}}}
    cloads::Dict{Int, Vector{TF}}
    facesets::Dict{String, Vector{Tuple{TI,TI}}}
    dloads::Dict{String, TF}
end
```

- `node_coords`: a vector of node coordinates.
- `celltype`: a cell type code in the INP convention 
- `cells`: a vector of cell connectivities
- `nodesets`: a dictionary mapping a node set name to a vector of node indices
- `cellsets`: a dictionary mapping a cell set name to a vector of cell indices
- `E`: Young's modulus
- `ν`: Poisson ratio
- `density`: physical density of the material
- `nodedbcs`: a dictionary mapping a node set name to a vector of tuples of type `Tuple{Int, Float64}` specifying a Dirichlet boundary condition on that node set. Each tuple in the vector specifies the local index of a constrained degree of freedom and its fixed value. A 3-dimensional field has 3 degrees of freedom per node for example. So the index can be 1, 2 or 3.
- `cloads`: a dictionary mapping a node index to a load vector on that node.
- `facesets`: a dictionary mapping a face set name to a vector of `Tuple{Int,Int}` tuples where each tuple is a face index. The first integer is the cell index where the face is and the second integer is the local face index in the cell according to the VTK convention.
- `dloads`: a dictionary of distributed loads mapping face set names to a normal traction load value.
"""
struct InpContent{dim, TF, N, TI}
    node_coords::Vector{NTuple{dim,TF}}
    celltype::String
    cells::Vector{NTuple{N,TI}}
    nodesets::Dict{String,Vector{TI}}
    cellsets::Dict{String,Vector{TI}}
    E::TF
    ν::TF
    density::TF
    nodedbcs::Dict{String, Vector{Tuple{TI,TF}}}
    cloads::Dict{Int, Vector{TF}}
    facesets::Dict{String, Vector{Tuple{TI,TI}}}
    dloads::Dict{String, TF}
end

const stopping_pattern = r"^\*[^\*]"

# Import
include(joinpath("FeatureExtractors", "extract_cells.jl"))
include(joinpath("FeatureExtractors", "extract_cload.jl"))
include(joinpath("FeatureExtractors", "extract_dbcs.jl"))
include(joinpath("FeatureExtractors", "extract_dload.jl"))
include(joinpath("FeatureExtractors", "extract_material.jl"))
include(joinpath("FeatureExtractors", "extract_nodes.jl"))
include(joinpath("FeatureExtractors", "extract_set.jl"))
include(joinpath("inp_to_ferrite.jl"))

function extract_inp(filepath_with_ext)
    file = open(filepath_with_ext, "r")
    
    local node_coords
    local celltype, cells, offset
    nodesets = Dict{String,Vector{Int}}()
    cellsets = Dict{String,Vector{Int}}()
    local E, mu
    nodedbcs = Dict{String, Vector{Tuple{Int,Float64}}}()
    cloads = Dict{Int, Vector{Float64}}()
    facesets = Dict{String, Vector{Tuple{Int,Int}}}()
    dloads = Dict{String, Float64}()
    density = 0. # Should extract from the file

    node_heading_pattern = r"\*Node\s*,\s*NSET\s*=\s*([^,]*)"
    cell_heading_pattern = r"\*Element\s*,\s*TYPE\s*=\s*([^,]*)\s*,\s*ELSET\s*=\s*([^,]*)"
    nodeset_heading_pattern = r"\*NSET\s*,\s*NSET\s*=\s*([^,]*)"
    cellset_heading_pattern = r"\*ELSET\s*,\s*ELSET\s*=\s*([^,]*)"
    material_heading_pattern = r"\*MATERIAL\s*,\s*NAME\s*=\s*([^\s]*)"
    boundary_heading_pattern = r"\*BOUNDARY"
    cload_heading_pattern = r"\*CLOAD"
    dload_heading_pattern = r"\*DLOAD"

    line = readline(file)
    local dim
    while !eof(file)
        m = match(node_heading_pattern, line)
        if m != nothing && m[1] == "Nall"
            node_coords, line = extract_nodes(file)
            dim = length(node_coords[1])
            continue
        end
        m = match(cell_heading_pattern, line)
        if m != nothing
            celltype = String(m[1])
            cellsetname = String(m[2]) 
            cells, offset, line = extract_cells(file)
            cellsets[cellsetname] = collect(1:length(cells))
            continue
        end
        m = match(nodeset_heading_pattern, line)
        if m != nothing
            nodesetname = String(m[1])
            line = extract_set!(nodesets, nodesetname, file)
            continue
        end
        m = match(cellset_heading_pattern, line)
        if m != nothing
            cellsetname = String(m[1])
            line = extract_set!(cellsets, cellsetname, file, offset)
            continue
        end
        m = match(material_heading_pattern, line)
        if m != nothing
            material_name = String(m[1])
            E, mu, line = extract_material(file)
            continue
        end
        m = match(boundary_heading_pattern, line)
        if m != nothing
            line = extract_nodedbcs!(nodedbcs, file)
            continue
        end
        m = match(cload_heading_pattern, line)
        if m != nothing
            line = extract_cload!(cloads, file, Val{dim})
            continue
        end
        m = match(dload_heading_pattern, line)
        if m != nothing
            line = extract_dload!(dloads, facesets, file, Val{dim}, offset)
            continue
        end
        line = readline(file)
    end

    close(file)

    return InpContent(node_coords, celltype, cells, nodesets, cellsets, E, mu, density, nodedbcs, cloads, facesets, dloads)
end

end
