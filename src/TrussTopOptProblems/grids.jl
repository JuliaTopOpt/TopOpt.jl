using ..TopOpt.TopOptProblems: AbstractGrid
const Vec = JuAFEM.Vec

# @params struct TrussGrid{xdim,N,M,C<:JuAFEM.Cell{xdim,N,M},T} <: AbstractGrid{xdim, T}
struct TrussGrid{xdim,T,N,M,TG<:JuAFEM.Grid{xdim,<:JuAFEM.Cell{xdim,N,M},T}} <: AbstractGrid{xdim, T}
    grid::TG
    white_cells::BitVector
    black_cells::BitVector
    constant_cells::BitVector
    crosssecs::Vector{TrussFEACrossSec{T}}
end
# nels::NTuple{dim, Int} # num of elements in x,y,z direction in the ground mesh, not applicable to truss
# sizes::NTuple{dim, T}  # length of the ground mesh in x,y,z direction, not applicaiton to truss
# corners::NTuple{2, Vec{dim, T}} # corner of the ground mesh

nnodespercell(::TrussGrid{xdim,T,N,M}) where {xdim,T,N,M} = N
nfacespercell(::TrussGrid{xdim,T,N,M}) where {xdim,T,N,M} = M
nnodes(cell::Type{JuAFEM.Cell{dim,N,M}}) where {dim, N, M} = N
nnodes(cell::JuAFEM.Cell) = nnodes(typeof(cell))
JuAFEM.getncells(tg::TrussGrid) = JuAFEM.getncells(tg.grid)

function TrussGrid(node_points::Dict{iT, SVector{xdim, T}}, elements::Dict{iT, Tuple{iT, iT}}, 
        boundary::Dict{iT, SVector{xdim, fT}}; crosssecs=TrussFEACrossSec{T}(1.0)) where {xdim, T, iT, fT}
    # ::Type{Val{CellType}}, 
    # if CellType === :Linear
    #     geoshape = Line
    # else
    #     @assert false "not implemented"
    #     # geoshape = QuadraticQuadrilateral
    # end

    grid = _LinearTrussGrid(node_points, elements, boundary)
    ncells = getncells(grid)
    if crosssecs isa Vector
        @assert length(crosssecs) == ncells
        crosssecs = convert(Vector{TrussFEACrossSec{T}}, crosssecs)
    elseif crosssecs isa TrussFEACrossSec
        crosssecs = [TrussFEACrossSec{T}(crosssecs) for i=1:ncells]
    else
        error("Invalid crossecs: $(crossecs)")
    end
    return TrussGrid(grid, falses(ncells), falses(ncells), falses(ncells), crosssecs)
end

function _LinearTrussGrid(node_points::Dict{iT, SVector{xdim, T}}, elements::Dict{iT, Tuple{iT, iT}}, 
        boundary::Dict{iT, SVector{xdim, fT}}) where {xdim, T, iT, fT}
    n_nodes = length(node_points)

    # * Generate cells, Line2d or Line3d
    CellType = Cell{xdim,2,2}
    cells = Vector{CellType}(undef, length(elements))
    for (e_ind, element) in elements
        cells[e_ind] = CellType((element...,))
    end

    # * Generate nodes
    nodes = Vector{Node{xdim,T}}(undef, length(node_points))
    for (n_id, node_point) in node_points
        nodes[n_id] = Node((node_point...,))
    end

    # ? not sure if we need to define boundary matrix in truss problems
    # # * label boundary (cell, face)
    # cell_from_node = node_neighbors(cells)
    # boundary_conditions = Tuple{Int,Int}[]
    # for (v, _) in boundary
    #     for c in cell_from_node[v]
    #         # TODO this is not correct, v should be 1 or 2
    #         push!(boundary_conditions, (c, v))
    #     end
    # end
    # boundary_matrix = JuAFEM.boundaries_to_sparse(boundary_conditions)

    # * label loaded facesets
    # # Cell face sets
    # facesets = Dict("left"  => Set{Tuple{Int,Int}}([boundary[1]]),
    #                 "right" => Set{Tuple{Int,Int}}([boundary[2]]))

    return Grid(cells, nodes)
    # , boundary_matrix=boundary_matrix
    # return Grid(cells, nodes, facesets=facesets, 
    #     boundary_matrix=boundary_matrix)
end

function Base.show(io::Base.IO, mime::MIME"text/plain", tg::TrussGrid)
    println(io, "TrussGrid:")
    print(io, "\t-")
    Base.show(io, mime, tg.grid)
    println(io,"")
    print(io, "\t-")
    println(io, "white cells:T|$(sum(tg.white_cells))|, black cells:T|$(sum(tg.black_cells))|, const cells:T|$(sum(tg.constant_cells))|")
end

"""
Compute a dict for querying connected cells (elements) to a given node index

Returns: dict (node_idx => [cell_idx, ...])
"""
function node_neighbors(cells)
    # Contains the elements that each node contain
    cell_from_node = Dict{Int, Set{Int}}()
    for (cellid, cell) in enumerate(cells)
        for v in cell.nodes
            if !haskey(cell_from_node, v)
                cell_from_node[v] = Set{Int}()
            end
            push!(cell_from_node[v], cellid)
        end
    end
    cell_from_node
end

################################

function Base.show(io::Base.IO, ::MIME"text/plain", grid::JuAFEM.Grid{xdim, JuAFEM.Cell{xdim,2,2}}) where {xdim}
    print(io, "$(typeof(grid)) with $(getncells(grid)) $(extra_celltypes[eltype(grid.cells)]) cells and $(getnnodes(grid)) nodes")
end

const extra_celltypes = Dict{DataType, String}(JuAFEM.Cell{2,2,2}  => "Line2D", 
                                               JuAFEM.Cell{3,2,2}  => "Line3D")