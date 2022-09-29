using ..TopOpt.TopOptProblems: AbstractGrid
const Vec = Ferrite.Vec

# @params struct TrussGrid{xdim,N,M,C<:Ferrite.Cell{xdim,N,M},T} <: AbstractGrid{xdim, T}
struct TrussGrid{xdim,T,N,M,TG<:Ferrite.Grid{xdim,<:Ferrite.Cell{xdim,N,M},T}} <:
       AbstractGrid{xdim,T}
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
nnodes(cell::Type{Ferrite.Cell{xdim,N,M}}) where {xdim,N,M} = N
nnodes(cell::Ferrite.Cell) = nnodes(typeof(cell))
Ferrite.getncells(tg::TrussGrid) = Ferrite.getncells(tg.grid)

function TrussGrid(
    node_points::Matrix{T},
    elements::Matrix{iT};
    crosssecs = TrussFEACrossSec{T}(1.0),
) where {xdim,T,iT,fT}
    grid = _LinearTrussGrid(node_points, elements)
    ncells = getncells(grid)
    if crosssecs isa Vector
        @assert length(crosssecs) == ncells
        crosssecs = convert(Vector{TrussFEACrossSec{T}}, crosssecs)
    elseif crosssecs isa TrussFEACrossSec
        crosssecs = [convert(TrussFEACrossSec{T}, crosssecs) for i = 1:ncells]
    else
        error("Invalid crossecs: $(crossecs)")
    end
    return TrussGrid(grid, falses(ncells), falses(ncells), falses(ncells), crosssecs)
end

function TrussGrid(
    node_points::Dict{iT,SVector{xdim,T}},
    elements::Dict{iT,Tuple{iT,iT}};
    crosssecs = TrussFEACrossSec{T}(1.0),
) where {xdim,T,iT,fT}
    grid = _LinearTrussGrid(node_points, elements)
    ncells = getncells(grid)
    if crosssecs isa Vector
        @assert length(crosssecs) == ncells
        crosssecs = convert(Vector{TrussFEACrossSec{T}}, crosssecs)
    elseif crosssecs isa TrussFEACrossSec
        crosssecs = [convert(TrussFEACrossSec{T}, crosssecs) for i = 1:ncells]
    else
        error("Invalid crossecs: $(crossecs)")
    end
    return TrussGrid(grid, falses(ncells), falses(ncells), falses(ncells), crosssecs)
end

function _LinearTrussGrid(node_points::Matrix{T}, elements::Matrix{iT}) where {T,iT}
    xdim, n_nodes = size(node_points)
    nconnect, n_elems = size(elements)

    # * Generate cells, Line2d or Line3d
    @assert xdim ∈ [2, 3]
    @assert nconnect == 2
    CellType = xdim == 2 ? Line2D : Line3D
    cells = Vector{CellType}(undef, n_elems)
    for e_ind = 1:n_elems
        cells[e_ind] = CellType((elements[:, e_ind]...,))
    end

    # * Generate nodes
    nodes = Vector{Node{xdim,T}}(undef, n_nodes)
    for n_id = 1:n_nodes
        nodes[n_id] = Node((node_points[:, n_id]...,))
    end

    return Grid(cells, nodes)
end

function _LinearTrussGrid(
    node_points::Dict{iT,SVector{xdim,T}},
    elements::Dict{iT,Tuple{iT,iT}},
) where {xdim,T,iT,fT}
    n_nodes = length(node_points)

    # * Generate cells, Line2d or Line3d
    @assert xdim ∈ [2, 3]
    CellType = xdim == 2 ? Line2D : Line3D
    cells = Vector{CellType}(undef, length(elements))
    for (e_ind, element) in elements
        cells[e_ind] = CellType((element...,))
    end

    # * Generate nodes
    nodes = Vector{Node{xdim,T}}(undef, length(node_points))
    for (n_id, node_point) in node_points
        nodes[n_id] = Node((node_point...,))
    end

    return Grid(cells, nodes)
end

################################

function Base.show(io::Base.IO, mime::MIME"text/plain", tg::TrussGrid)
    println(io, "TrussGrid:")
    print(io, "\t-")
    print(
        io,
        " $(typeof(tg.grid)) with $(getncells(tg.grid)) $(extra_celltypes[eltype(tg.grid.cells)]) cells and $(getnnodes(tg.grid)) nodes",
    )
    println(io, "")
    print(io, "\t-")
    return println(
        io,
        " white cells:T|$(sum(tg.white_cells))|, black cells:T|$(sum(tg.black_cells))|, const cells:T|$(sum(tg.constant_cells))|",
    )
end

const extra_celltypes = Dict{DataType,String}(Line2D => "Line2D", Line3D => "Line3D")
