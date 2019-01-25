module CheqFilters

using ..GPUUtils, ..Utilities, JuAFEM
using CuArrays, ..FEA, Statistics
using GPUArrays: GPUVector
import ..GPUUtils: whichdevice
using Parameters: @unpack
import CUDAdrv

export  AbstractCheqFilter,
        CheqFilter

const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

abstract type AbstractCheqFilter end

@params struct FilterMetadata
    cell_neighbouring_nodes
    cell_node_weights
end
@define_cu(FilterMetadata, :cell_neighbouring_nodes, :cell_node_weights)
GPUUtils.whichdevice(m::FilterMetadata) = whichdevice(m.cell_neighbouring_nodes)

@params struct CheqFilter{_filtering, T, TV <: AbstractVector{T}} <: AbstractCheqFilter
    filtering::Val{_filtering}
    metadata::FilterMetadata
    rmin::T
    nodal_grad::TV
    last_grad::TV
    cell_weights::TV
end
@define_cu(CheqFilter, :metadata, :nodal_grad, :last_grad, :cell_weights)
GPUUtils.whichdevice(c::CheqFilter) = whichdevice(c.nodal_grad)

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
    @unpack node_cells = problem.metadata
    TI = eltype(node_cells.offsets)

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
                    n_cells = node_cells[n]
                    for c in n_cells
                        next_cell_id = c[1]
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

function CheqFilter(::Val{filtering}, solver, args...) where {filtering}
    CheqFilter(Val(filtering), whichdevice(solver), solver, args...)
end

function CheqFilter(::Val{true}, ::CPU, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TI<:Integer, TS<:AbstractFEASolver}
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
    
    return CheqFilter(Val(true), metadata, rmin, nodal_grad, last_grad, cell_weights)
end

function CheqFilter(::Val{false}, ::CPU, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TS<:AbstractFEASolver, TI<:Integer}
    metadata = FilterMetadata(T, TI)
    nodal_grad = T[]
    last_grad = T[]
    cell_weights = T[]
    return CheqFilter(Val(false), metadata, TI(0), nodal_grad, last_grad, cell_weights)
end

function (cf::CheqFilter{true, T})(grad, vars, elementinfo) where {T}
    cf.rmin <= 0 && return
    @unpack nodal_grad, cell_weights, metadata = cf
    @unpack black, white, varind, cellvolumes, cells = elementinfo
    @unpack cell_neighbouring_nodes, cell_node_weights = metadata
    node_cells = elementinfo.metadata.node_cells

    update_nodal_grad!(nodal_grad, node_cells, cell_weights, cells, cellvolumes, black, white, varind, grad)
    normalize_grad!(nodal_grad, cell_weights)
    update_grad!(grad, black, white, varind, cell_neighbouring_nodes, cell_node_weights, nodal_grad)
end

(cf::CheqFilter{false})(args...) = nothing

function update_nodal_grad!(nodal_grad::AbstractVector, node_cells, cell_weights, cells, cellvolumes, black, white, varind, grad)
    T = eltype(nodal_grad)
    for n in 1:length(nodal_grad)
        nodal_grad[n] = zero(T)
        cell_weights[n] = zero(T)
        r = node_cells.offsets[n]:node_cells.offsets[n+1]-1
        for i in r
            c = node_cells.values[i][1]
            if black[c] || white[c]
                continue
            end
            ind = varind[c]
            w = cellvolumes[c]
            cell_weights[n] += w
            nodal_grad[n] += w * grad[ind]
        end
    end
    return
    #=
function update_nodal_grad!(nodal_grad::AbstractVector, cell_weights, cells, cellvolumes, black, white, varind, grad)
    T = eltype(nodal_grad)
    nodal_grad .= zero(T)
    cell_weights .= 0
    @inbounds for i in 1:length(cells)
        if black[i] || white[i]
            continue
        end
        nodes = cells[i].nodes
        for n in nodes
            ind = varind[i]
            w = cellvolumes[i]
            cell_weights[n] += w
            nodal_grad[n] += w*grad[ind]
        end
    end
    =#
end

function update_nodal_grad!(nodal_grad::GPUVector, node_cells, args...)
    T = eltype(nodal_grad)
    allargs = (nodal_grad, node_cells.offsets, node_cells.values, args...)
    callkernel(dev, cheq_kernel1, allargs)
    CUDAdrv.synchronize(ctx)
    return
end

function cheq_kernel1(nodal_grad, node_cells_offsets, node_cells_values, cell_weights, 
        cells, cellvolumes, black, white, varind, grad)
    T = eltype(nodal_grad)
    n = @thread_global_index()
    offset = @total_threads()
    while n <= length(nodal_grad)
        nodal_grad[n] = zero(T)
        cell_weights[n] = zero(T)
        r = node_cells_offsets[n]:node_cells_offsets[n+1]-1
        for i in r
            c = node_cells_values[i][1]
            if black[c] || white[c]
                continue
            end
            ind = varind[c]
            w = cellvolumes[c]
            cell_weights[n] += w
            nodal_grad[n] += w * grad[ind]
        end
        n += offset
    end
end


function normalize_grad!(nodal_grad::AbstractVector, cell_weights)
    for n in 1:length(nodal_grad)
        if cell_weights[n] > 0
            nodal_grad[n] /= cell_weights[n]
        end
    end
end
function normalize_grad!(nodal_grad::GPUVector, cell_weights)
    T = eltype(nodal_grad)
    args = (nodal_grad, cell_weights)
    callkernel(dev, cheq_kernel2, args)
    CUDAdrv.synchronize(ctx)
    return
end
function cheq_kernel2(nodal_grad, cell_weights)
    T = eltype(nodal_grad)
    n = @thread_global_index()
    offset = @total_threads()
    while n <= length(nodal_grad)
        w = cell_weights[n]
        w = ifelse(w > 0, w, one(T))
        nodal_grad[n] /= w
        n += offset
    end
    return
end


function update_grad!(grad::AbstractVector, black, white, varind, cell_neighbouring_nodes, cell_node_weights, nodal_grad)
    @inbounds for i in 1:length(black)
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

function update_grad!(grad::GPUVector, black, white, varind, cell_neighbouring_nodes, cell_node_weights, nodal_grad)
    T = eltype(grad)
    allargs = (grad, black, white, varind, cell_neighbouring_nodes.offsets, cell_neighbouring_nodes.values, cell_node_weights.values, nodal_grad)
    callkernel(dev, cheq_kernel3, allargs)
    CUDAdrv.synchronize(ctx)
    return
end
function cheq_kernel3(grad, black, white, varind, cell_neighbouring_nodes_offsets, cell_neighbouring_nodes_values, cell_node_weights_values, nodal_grad)
    T = eltype(nodal_grad)
    i = @thread_global_index()
    offset = @total_threads()
    while i <= length(black)
        if black[i] || white[i]
            continue
        end
        ind = varind[i]
        r = cell_neighbouring_nodes_offsets[ind]:cell_neighbouring_nodes_offsets[ind+1]-1
        length(r) == 0 && continue
        grad[ind] = zero(T)
        sum_weights = zero(T)
        for linear_ind in r
            w = cell_node_weights_values[linear_ind]
            sum_weights += w
            node_ind = cell_neighbouring_nodes_values[linear_ind]
            grad[ind] += nodal_grad[node_ind] * w
        end
        grad[ind] /= sum_weights
        i += offset
    end

    return
end

end
