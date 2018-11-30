abstract type AbstractCheqFilter end

struct RaggedArray{TO, TV}
    offsets::TO
    values::TV
end
whichdevice(ra::RaggedArray) = whichdevice(ra.offsets)

function RaggedArray(vv::Vector{Vector{T}}) where T
    offsets = [1; 1 .+ accumulate(+, collect(length(v) for v in vv))]
    values = zeros(T, offsets[end]-1)
    for (i, v) in enumerate(vv)
        r = offsets[i]:offsets[i+1]-1
        values[r] .= v
    end
    RaggedArray(offsets, values)
end
@define_cu(RaggedArray, :offsets, :values)

function Base.getindex(ra::RaggedArray, i)
    @assert 1 <= j < length(ra.offsets)
    r = ra.offsets[i]:ra.offsets[i+1]-1
    @assert 1 <= r[1] && r[end] <= length(ra.values)
    return @view ra.values[r]
end
function Base.getindex(ra::RaggedArray, i, j)
    @assert 1 <= j < length(ra.offsets)
    r = ra.offsets[j]:ra.offsets[j+1]-1
    @assert 1 <= i <= length(r)
    return ra.values[r[i]]
end
function Base.setindex!(ra::RaggedArray, v, i, j)
    @assert 1 <= j < length(ra.offsets)
    r = ra.offsets[j]:ra.offsets[j+1]-1
    @assert 1 <= i <= length(r)
    ra.values[r[i]] = v
end

struct FilterMetadata{TN,TW}
    cell_neighbouring_nodes::TN
    cell_node_weights::TW
end
@define_cu(FilterMetadata, :cell_neighbouring_nodes, :cell_node_weights)
whichdevice(m::FilterMetadata) = whichdevice(m.cell_neighbouring_nodes)

struct CheqFilter{filtering, T, TV<:AbstractVector{T}, TM<:FilterMetadata, TS<:AbstractFEASolver} <: AbstractCheqFilter
    filtering::Val{filtering}
    solver::TS
    metadata::TM
    rmin::T
    nodal_grad::TV
    last_grad::TV
    cell_weights::TV
end
@define_cu(CheqFilter, :metadata, :nodal_grad, :last_grad, :cell_weights)
whichdevice(c::CheqFilter) = whichdevice(c.nodal_grad)

function FilterMetadata(::Type{T}, ::Type{TI}) where {T, TI}
    cell_neighbouring_nodes = Vector{TI}[]
    cell_node_weights = Vector{T}[]

    return FilterMetadata(cell_neighbouring_nodes, cell_node_weights)
end

function FilterMetadata(solver, rmin::T, ::Type{TI}) where {T, TI}
    problem = solver.problem
    cell_neighbouring_nodes, cell_node_weights = get_neighbour_info(problem, rmin)
    return FilterMetadata(RaggedArray(cell_neighbouring_nodes), RaggedArray(cell_node_weights))
end

function get_neighbour_info(problem, rmin::T) where {T}
    @unpack black, white, varind = problem
    dh = problem.ch.dh
    grid = dh.grid
    @unpack node_cells, node_cells_offset = problem.metadata
    TI = eltype(node_cells[1])

    all_neighbouring_nodes = Vector{TI}[]
    all_node_weights = Vector{T}[]
    sizehint!(all_neighbouring_nodes, getncells(grid))
    sizehint!(all_node_weights, getncells(grid))

    cells_to_traverse = TI[]
    neighbouring_nodes = TI[]
    node_weights = T[]
    visited_cells = Set{Int}()
    for cell in CellIterator(dh)
        current_cell_id = cell.current_cellid.x
        if black[current_cell_id] || white[current_cell_id]
            continue
        end
        empty!(cells_to_traverse)
        empty!(visited_cells)
        empty!(neighbouring_nodes)
        empty!(node_weights)

        center = Node(mean(cell.coords).data)

        push!(cells_to_traverse, current_cell_id)
        push!(visited_cells, current_cell_id)
        while !isempty(cells_to_traverse)
            # Takes first element and removes it -> breadth first traversal
            cell_id = popfirst!(cells_to_traverse)
            for n in grid.cells[cell_id].nodes
                node = getnodes(grid, n)
                dist = norm(node.x - center.x)
                if dist < rmin
                    push!(neighbouring_nodes, n)
                    push!(node_weights, max(rmin-dist, zero(T)))
                    r = node_cells_offset[n] : node_cells_offset[n+1] - 1
                    for j in r
                        next_cell_id = node_cells[j][1]
                        if !(next_cell_id in visited_cells)
                            push!(cells_to_traverse, next_cell_id)
                            push!(visited_cells, next_cell_id)
                        end
                    end
                end
            end
        end
        push!(all_neighbouring_nodes, copy(neighbouring_nodes))
        push!(all_node_weights, copy(node_weights))
    end

    return all_neighbouring_nodes, all_node_weights
end

CheqFilter{true}(args...) = CheqFilter(Val(true), args...)
CheqFilter{false}(args...) = CheqFilter(Val(false), args...)

function CheqFilter(::Val{true}, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TI<:Integer, TS<:AbstractFEASolver}
    metadata = FilterMetadata(solver, rmin, TI)
    TM = typeof(metadata)
    problem = solver.problem
    grid = problem.ch.dh.grid
    nnodes = getnnodes(grid)
    nodal_grad = zeros(T, nnodes)
    TV = typeof(nodal_grad)

    black = problem.black
    white = problem.white
    nel = length(black)
    nfc = sum(black) + sum(white)
    last_grad = zeros(T, nel-nfc)

    cell_weights = zeros(T, nnodes)
    
    return CheqFilter(Val(true), solver, metadata, rmin, nodal_grad, last_grad, cell_weights)
end

function CheqFilter(::Val{false}, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TS<:AbstractFEASolver, TI<:Integer}
    metadata = FilterMetadata(T, TI)
    nodal_grad = T[]
    last_grad = T[]
    cell_weights = T[]
    return CheqFilter(Val(false), solver, metadata, TI(0), nodal_grad, last_grad, cell_weights)
end

function (cf::CheqFilter{true, T})(grad) where {T}
    @unpack solver, nodal_grad, cell_weights, metadata = cf
    @unpack vars, problem, elementinfo = solver
    cell_volumes = elementinfo.cellvolumes
    @unpack cell_neighbouring_nodes, cell_node_weights = metadata
    @unpack black, white, varind = metadata
    cells = problem.ch.dh.grid.cells

    update_nodal_grad!(nodal_grad, cells, )
    nodal_grad .= zero(T)
    cell_weights .= 0
    @inbounds for i in 1:length(cells)
        if black[i] || white[i]
            continue
        end
        nodes = cells[i].nodes
        for n in nodes
            ind = varind[i]
            w = cell_volumes[i]
            cell_weights[n] += w
            nodal_grad[n] += w*grad[ind]
        end
    end
    for n in 1:length(nodal_grad)
        if cell_weights[n] > 0
            nodal_grad[n] /= cell_weights[n]
        end
    end
    
    @inbounds for i in 1:length(cells)
        if black[i] || white[i]
            continue
        end
        ind = varind[i]
        nodes = cell_neighbouring_nodes[ind]
        if length(nodes) == 0
            continue
        end
        weights = cell_node_weights[ind]
        grad[ind] = dot(view(nodal_grad, nodes), weights) / sum(weights)
    end

    return
end

(cf::CheqFilter{false})(grad) = nothing
