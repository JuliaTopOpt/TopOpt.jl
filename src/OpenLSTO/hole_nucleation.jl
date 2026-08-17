# Hole-nucleation scheme for the compliance problem (a port of
# `projects/hole_creation/lsm_hole_insertion.hpp` and
# `projects/hole_creation/boundary_hole.hpp`). A secondary "hole" level-set
# function is built from the extrapolated nodal sensitivities; where it drops
# below zero in regions outside the narrow band, new holes are inserted into
# the primary level-set function and the result is reinitialized by the fast
# marching method.

# Material area enclosed by an arbitrary signed-distance field, computed on a
# copy of the node/element status flags so the mesh is left untouched (a port
# of `Boundary_hole::computeAreaFractions`).
function hole_area_fractions(mesh::LevelSetMesh, signed_distance::Vector{Float64})
    node_status = zeros(Int, mesh.nNodes)
    for i in eachindex(mesh.nodes)
        sd = signed_distance[i]
        node_status[i] = if abs(sd) < 1e-6
            NODE_BOUNDARY
        elseif sd < 0
            NODE_OUTSIDE
        else
            NODE_INSIDE
        end
    end

    element_status = zeros(Int, mesh.nElements)
    for i in eachindex(mesh.elements)
        inside = 0
        outside = 0
        for j in 1:4
            s = node_status[mesh.elements[i].nodes[j]]
            if (s & NODE_INSIDE) != 0
                inside += 1
            elseif (s & NODE_OUTSIDE) != 0
                outside += 1
            end
        end
        element_status[i] = if outside == 0
            ELEMENT_INSIDE
        elseif inside == 0
            ELEMENT_OUTSIDE
        else
            ELEMENT_NONE
        end
    end

    area = 0.0
    for i in eachindex(mesh.elements)
        if (element_status[i] & ELEMENT_INSIDE) != 0
            area += 1.0
        elseif (element_status[i] & ELEMENT_OUTSIDE) != 0
            area += 0.0
        else
            area += hole_cut_area(mesh, mesh.elements[i], node_status, element_status[i])
        end
    end
    return area
end

# Polygon area of the inside region of a cut element, using only the node
# status flags (no boundary segments; a port of `Boundary_hole::cutArea`).
function hole_cut_area(
    mesh::LevelSetMesh, element, node_status::Vector{Int}, element_status::Int
)
    status = (element_status & ELEMENT_CENTRE_OUTSIDE) != 0 ? NODE_OUTSIDE : NODE_INSIDE
    vertices = Coord[]
    for i in 1:4
        node = element.nodes[i]
        if (node_status[node] & status) != 0
            push!(vertices, mesh.nodes[node].coord)
        elseif (node_status[node] & NODE_BOUNDARY) != 0
            n1 = element.nodes[i == 4 ? 1 : i + 1]
            n2 = element.nodes[i == 1 ? 4 : i - 1]
            if (node_status[n1] & NODE_INSIDE) != 0 && (node_status[n2] & NODE_INSIDE) != 0
                push!(vertices, mesh.nodes[node].coord)
            end
        end
    end
    return polygon_area(vertices, element.coord)
end

# Identify the nodes and elements available for hole insertion (a port of
# `hole_map`). A node is available when its signed distance is outside the
# narrow band (>= `l_band * h`) and it is active or not fixed. Returns the
# number of available elements and the node/element availability masks.
function hole_map(mesh::LevelSetMesh, level_set::LevelSet, h::Real, l_band::Real)
    nb = l_band * h
    h_index = zeros(Bool, mesh.nNodes)
    for i in eachindex(mesh.nodes)
        node = mesh.nodes[i]
        if level_set.signedDistance[i] >= nb && (node.isActive || !node.isFixed)
            h_index[i] = true
        end
    end

    h_elem = zeros(Bool, mesh.nElements)
    count = 0
    for iel in eachindex(mesh.elements)
        available = false
        for ind in 1:4
            node = mesh.elements[iel].nodes[ind]
            if level_set.signedDistance[node] >= nb &&
                (mesh.nodes[node].isActive || !mesh.nodes[node].isFixed)
                available = true
                break
            end
        end
        if available
            count += 1
            h_elem[iel] = true
        end
    end
    return count, h_index, h_elem
end

# Update the secondary hole level-set function in place (a port of
# `get_h_lsf`): for each available node, subtract the sensitivity weighted by
# the Lagrange multipliers.
function get_h_lsf!(
    h_index::Vector{Bool},
    h_nsens::Vector{Vector{Float64}},
    gammas::Vector{Float64},
    h_lsf::Vector{Float64},
)
    for inode in eachindex(h_index)
        if h_index[inode]
            for j in eachindex(gammas)
                h_lsf[inode] -= gammas[j] * h_nsens[inode][j]
            end
        end
    end
    return h_lsf
end
