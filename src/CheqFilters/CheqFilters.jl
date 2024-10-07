module CheqFilters

using ..Utilities, Ferrite
using ..FEA, Statistics
using ..TopOpt: TopOpt, ElementFEAInfo, PseudoDensities
import ..TopOpt: PENALTY_BEFORE_INTERPOLATION
using Parameters: @unpack
using SparseArrays, LinearAlgebra
using ForwardDiff
using Nonconvex: Nonconvex
using ChainRulesCore

export AbstractCheqFilter,
    SensFilter,
    DensityFilter,
    ProjectedDensityFilter,
    AbstractSensFilter,
    AbstractDensityFilter

abstract type AbstractCheqFilter end
abstract type AbstractSensFilter <: AbstractCheqFilter end
abstract type AbstractDensityFilter <: AbstractCheqFilter end

struct FilterMetadata{TC1,TC2}
    cell_neighbouring_nodes::TC1
    cell_node_weights::TC2
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::FilterMetadata)
    return println("TopOpt filter metadata")
end

function FilterMetadata(::Type{T}, ::Type{TI}) where {T,TI}
    cell_neighbouring_nodes = Vector{TI}[]
    cell_node_weights = Vector{T}[]

    return FilterMetadata(cell_neighbouring_nodes, cell_node_weights)
end

function FilterMetadata(solver, rmin::T, ::Type{TI}) where {T,TI}
    problem = solver.problem
    cell_neighbouring_nodes, cell_node_weights = get_neighbour_info(problem, rmin)
    return FilterMetadata(
        RaggedArray(cell_neighbouring_nodes), RaggedArray(cell_node_weights)
    )
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
                    push!(node_weights, max(rmin - dist, zero(T)))
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

include("sens_filter.jl")
include("density_filter.jl")

end
