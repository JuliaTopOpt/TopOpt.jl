"""
    struct Metadata
        cell_dofs
        dof_cells
        node_cells
        node_dofs
    end

An instance of the `Metadata` type stores ... information such as:
- `cell_dofs`: a `ndof_per_cell` x `ncells` matrix that maps `[localdofidx, cellidx]` into dof idx
    cell_dofs[j,cellidx] = j-th nodal dof of cell `cellidx`
- `dof_cells`: a `ncells_per_dof` x `ndofs` `RaggedArray` of `Tuple{Int, Int}` that maps each dof to its cells and its local dof idx in each cell
    dof_cells[dofidx] = [(cellidx, cell's local dof idx), ...]
- `node_cells`: a `ncells_per_node` x `nnodes` `RaggedArray` of `Tuple{Int, Int}` that maps each node to its cells and its local idx in each cell
    node_cells[node_idx] = [(cellidx, cell's local node idx), ...]
- `node_dofs`: a `ndofspernode x nnodes` Matrix that maps `[localdofidx, node_idx]` into dof indices
    node_dofs[j,nodeidx] = j-th dof of node `nodeidx`
"""
@params struct Metadata
    cell_dofs
    dof_cells
    #node_first_cells::TTupleVec
    node_cells
    node_dofs
end

function Metadata(dh::DofHandler{dim}) where dim
    cell_dofs = get_cell_dofs_matrix(dh)
    dof_cells = get_dof_cells_matrix(dh, cell_dofs)
    #node_first_cells = get_node_first_cells(dh)
    node_cells = get_node_cells(dh)
    node_dofs = get_node_dofs(dh)

    meta = Metadata(cell_dofs, dof_cells, node_cells, node_dofs)
end

"""
Returns a `ndof_per_cell` x `ncells` matrix that maps `[localdofidx, cellidx]` into nof index
"""
function get_cell_dofs_matrix(dh)
    cell_dofs = zeros(Int, ndofs_per_cell(dh), getncells(dh.grid))
    for i in 1:size(cell_dofs, 2)
        r = dh.cell_dofs_offset[i]:dh.cell_dofs_offset[i+1]-1
        for j in 1:length(r)
            cell_dofs[j,i] = dh.cell_dofs[r[j]]
        end
    end
    cell_dofs
end

"""
Inputs
======
- `dh`: DofHandler
- `cell_dofs`: matrix `ndof_per_cell` x `ncells`: cell_dofs[j,i] = j-th nodal dof of cell i

Returns
=======
- a RaggedArray that maps dof into cell and its local dof idx
    `ndof` x (Vector of Tuple{Int, Int})
        dof_cells_matrix[dofidx] = [(cellidx, cell's local dof idx), ...]
"""
function get_dof_cells_matrix(dh, cell_dofs)
    dof_cells_vecofvecs = [Vector{Tuple{Int,Int}}() for i in 1:ndofs(dh)]
    l = 0
    for cellidx in 1:size(cell_dofs, 2)
        for localidx in 1:size(cell_dofs, 1)
            dofidx = cell_dofs[localidx, cellidx]
            push!(dof_cells_vecofvecs[dofidx], (cellidx, localidx))
            l += 1
        end
    end

    return RaggedArray(dof_cells_vecofvecs)    
end

function get_node_first_cells(dh)
    node_first_cells = fill((0,0), getnnodes(dh.grid))
    visited = falses(getnnodes(dh.grid))
    for cellidx in 1:getncells(dh.grid)
        for (local_node_idx, global_node_idx) in enumerate(dh.grid.cells[cellidx].nodes)
            if !visited[global_node_idx]
                visited[global_node_idx] = true
                node_first_cells[global_node_idx] = (cellidx, local_node_idx)
            end
        end
    end
    return node_first_cells
end

"""
Returns
=======
- A RaggedArray that maps dof index into connected cell indices
    `ndof` x (Vector of Tuple{Int, Int})
        dof_cells_matrix[nodeidx] = [(cellidx, cell's local nodal idx), ...]
"""
function get_node_cells(dh)
    node_cells_vecofvecs = [Vector{Tuple{Int,Int}}() for i in 1:ndofs(dh)]
    l = 0
    for (cellidx, cell) in enumerate(CellIterator(dh))
        for (localidx, nodeidx) in enumerate(cell.nodes)
            push!(node_cells_vecofvecs[nodeidx], (cellidx, localidx))
            l += 1
        end
    end
    return RaggedArray(node_cells_vecofvecs)
end

node_field_offset(dh, f) = sum(view(dh.field_dims, 1:f-1))

"""
Returns
=======
- A `ndofspernode x nnodes` Matrix that maps `[localdofidx, node_idx]` into dof indices
"""
function get_node_dofs(dh::DofHandler)
    ndofspernode = sum(dh.field_dims)
    nfields = length(dh.field_dims)
    nnodes = getnnodes(dh.grid)
    interpol_points = ndofs_per_cell(dh)
    _celldofs = fill(0, ndofs_per_cell(dh))
    node_dofs = zeros(Int, ndofspernode, nnodes)
    visited = falses(nnodes)
    for field in 1:nfields
        field_dim = dh.field_dims[field]
        node_offset = node_field_offset(dh, field)
        offset = Ferrite.field_offset(dh, dh.field_names[field])
        for (cellidx, cell) in enumerate(dh.grid.cells)
            celldofs!(_celldofs, dh, cellidx) # update the dofs for this cell
            for idx in 1:min(interpol_points, length(cell.nodes))
                node = cell.nodes[idx]
                if !visited[node]
                    noderange = (offset + (idx-1)*field_dim + 1):(offset + idx*field_dim) # the dofs in this node
                    for i in 1:field_dim
                        node_dofs[node_offset+i,node] = _celldofs[noderange[i]]
                    end
                    visited[node] = true
                end
            end
        end
    end
    return node_dofs
end
