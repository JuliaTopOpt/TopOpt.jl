# Discretised boundary and element area fractions (a port of
# `M2DO_LSM/src/boundary.cpp`). The zero contour of the level set is
# triangulated per element with marching squares, producing boundary points,
# boundary segments, and a material area fraction for each element.

"""
    LevelSetBoundary(level_set)

The discretized boundary of a [`LevelSet`](@ref). [`discretise!`](@ref)
finds the boundary points where the zero contour crosses grid edges (marching
squares) and [`compute_area_fractions!`](@ref) computes the material area
fraction of every cell.
"""
mutable struct LevelSetBoundary
    levelSet::LevelSet
    points::Vector{LevelSetBoundaryPoint}
    segments::Vector{LevelSetBoundarySegment}
    length::Float64
    area::Float64
end
function LevelSetBoundary(ls::LevelSet)
    return LevelSetBoundary(
        ls, LevelSetBoundaryPoint[], LevelSetBoundarySegment[], 0.0, 0.0
    )
end

function initialise_point(boundary::LevelSetBoundary, coord::Coord, size_lambdas::Int)
    ls = boundary.levelSet
    mesh = ls.mesh
    point = LevelSetBoundaryPoint(
        coord,
        Coord(0.0, 0.0),
        0.0,
        0.0,
        0.0,
        0.0,
        false,
        false,
        Int[],
        Int[],
        zeros(size_lambdas),
    )
    point.negativeLimit = -ls.moveLimit
    point.positiveLimit = ls.moveLimit

    min_x = min(coord.x, mesh.width - coord.x)
    min_y = min(coord.y, mesh.height - coord.y)
    min_boundary = min(min_x, min_y)
    if min_boundary < ls.moveLimit
        point.negativeLimit = -min_boundary
        if min_boundary < 1e-6
            point.isDomain = true
        end
    end

    node = get_closest_node(mesh, coord)
    if mesh.nodes[node].isDomain
        dx = mesh.nodes[node].coord.x - coord.x
        dy = mesh.nodes[node].coord.y - coord.y
        if abs(dx) < 1e-6 && abs(dy) < 1e-6
            point.isDomain = true
            point.negativeLimit = 0.0
        else
            d = sqrt(dx * dx + dy * dy)
            if -d > point.negativeLimit
                point.negativeLimit = -d
            end
        end
    end
    return point
end

function add_point!(boundary::LevelSetBoundary, coord::Coord, size_lambdas::Int)
    push!(boundary.points, initialise_point(boundary, coord, size_lambdas))
    return length(boundary.points)
end

# Determine the node and element status flags from the signed distance.
function compute_mesh_status!(boundary::LevelSetBoundary, sd::Vector{Float64})
    mesh = boundary.levelSet.mesh
    for i in eachindex(mesh.nodes)
        empty!(mesh.nodes[i].boundaryPoints)
        if abs(sd[i]) < 1e-6
            mesh.nodes[i].status = NODE_BOUNDARY
        elseif sd[i] < 0
            mesh.nodes[i].status = NODE_OUTSIDE
        else
            mesh.nodes[i].status = NODE_INSIDE
        end
    end
    for i in eachindex(mesh.elements)
        element = mesh.elements[i]
        empty!(element.boundarySegments)
        tallyInside = 0
        tallyOutside = 0
        for j in 1:4
            status = mesh.nodes[element.nodes[j]].status
            if (status & NODE_INSIDE) != 0
                tallyInside += 1
            elseif (status & NODE_OUTSIDE) != 0
                tallyOutside += 1
            end
        end
        if tallyOutside == 0
            element.status = ELEMENT_INSIDE
        elseif tallyInside == 0
            element.status = ELEMENT_OUTSIDE
        else
            element.status = ELEMENT_NONE
        end
    end
end

# Work out the coordinates of a boundary point on an element edge and return
# the index of any previously added point at that location.
function is_added(boundary::LevelSetBoundary, node::Int, edge::Int, distance::Float64)
    mesh = boundary.levelSet.mesh
    nc = mesh.nodes[node].coord
    if edge == 0
        coord = Coord(nc.x + distance, nc.y)
    elseif edge == 1
        coord = Coord(nc.x, nc.y + distance)
    elseif edge == 2
        coord = Coord(nc.x - distance, nc.y)
    else
        coord = Coord(nc.x, nc.y - distance)
    end
    for index in mesh.nodes[node].boundaryPoints
        bp = boundary.points[index]
        if abs(coord.x - bp.coord.x) < 1e-6 && abs(coord.y - bp.coord.y) < 1e-6
            return index, coord
        end
    end
    return -1, coord
end

"""
    discretise!(boundary, size_lambdas)

Discretize the zero contour of the level set into boundary points and
segments with marching squares.
"""
function discretise!(boundary::LevelSetBoundary, size_lambdas::Int)
    ls = boundary.levelSet
    mesh = ls.mesh
    sd = ls.signedDistance
    empty!(boundary.points)
    empty!(boundary.segments)
    boundary.length = 0.0

    compute_mesh_status!(boundary, sd)

    for i in 1:(mesh.nElements)
        element = mesh.elements[i]
        if element.status == ELEMENT_OUTSIDE
            continue
        end

        nCut = 0
        bp_indices = zeros(Int, 4)

        for j in 0:3
            n1 = element.nodes[j + 1]
            n2 = element.nodes[(j == 3 ? 0 : j + 1) + 1]
            if mesh.nodes[n1].isActive || mesh.nodes[n2].isActive
                if (mesh.nodes[n1].status | mesh.nodes[n2].status) == NODE_CUT
                    d = sd[n1] / (sd[n1] - sd[n2])
                    index, coord = is_added(boundary, n1, j, d)
                    if index < 0
                        index = add_point!(boundary, coord, size_lambdas)
                        push!(mesh.nodes[n1].boundaryPoints, index)
                        push!(mesh.nodes[n2].boundaryPoints, index)
                    end
                    nCut += 1
                    bp_indices[nCut] = index
                elseif (mesh.nodes[n1].status & NODE_BOUNDARY) != 0 &&
                    (mesh.nodes[n2].status & NODE_BOUNDARY) != 0
                    index, coord = is_added(boundary, n1, 0, 0.0)
                    if index < 0
                        index = add_point!(boundary, coord, size_lambdas)
                        push!(mesh.nodes[n1].boundaryPoints, index)
                    end
                    start = index
                    index, coord = is_added(boundary, n2, 0, 0.0)
                    if index < 0
                        index = add_point!(boundary, coord, size_lambdas)
                        push!(mesh.nodes[n2].boundaryPoints, index)
                    end
                    segment = LevelSetBoundarySegment(start, index, i, 0.0, 0.0)
                    segment.length = segment_length(boundary, segment)
                    boundary.length += segment.length
                    push!(element.boundarySegments, length(boundary.segments) + 1)
                    push!(boundary.segments, segment)
                end
            end
        end

        if nCut == 2
            segment = LevelSetBoundarySegment(bp_indices[1], bp_indices[2], i, 0.0, 0.0)
            segment.length = segment_length(boundary, segment)
            boundary.length += segment.length
            push!(element.boundarySegments, length(boundary.segments) + 1)
            push!(boundary.segments, segment)
        elseif nCut == 1
            for j in 0:3
                node = element.nodes[j + 1]
                if (mesh.nodes[node].status & NODE_BOUNDARY) != 0
                    nAfter = element.nodes[(j == 3 ? 0 : j + 1) + 1]
                    nBefore = element.nodes[(j == 0 ? 3 : j - 1) + 1]
                    if (mesh.nodes[nAfter].status & NODE_OUTSIDE) != 0 ||
                        (mesh.nodes[nBefore].status & NODE_OUTSIDE) != 0
                        index, coord = is_added(boundary, node, 0, 0.0)
                        if index < 0
                            index = add_point!(boundary, coord, size_lambdas)
                            push!(mesh.nodes[node].boundaryPoints, index)
                        end
                        segment = LevelSetBoundarySegment(bp_indices[1], index, i, 0.0, 0.0)
                        segment.length = segment_length(boundary, segment)
                        boundary.length += segment.length
                        push!(element.boundarySegments, length(boundary.segments) + 1)
                        push!(boundary.segments, segment)
                    end
                end
            end
        elseif nCut == 4
            lsfSum = 0.0
            for j in 1:4
                lsfSum += sd[element.nodes[j]]
            end
            status = mesh.nodes[element.nodes[1]].status
            if ((status & NODE_INSIDE) != 0 && lsfSum > 0) ||
                ((status & NODE_OUTSIDE) != 0 && lsfSum < 0)
                seg1 = LevelSetBoundarySegment(bp_indices[1], bp_indices[2], i, 0.0, 0.0)
                seg2 = LevelSetBoundarySegment(bp_indices[3], bp_indices[4], i, 0.0, 0.0)
            else
                seg1 = LevelSetBoundarySegment(bp_indices[1], bp_indices[4], i, 0.0, 0.0)
                seg2 = LevelSetBoundarySegment(bp_indices[2], bp_indices[3], i, 0.0, 0.0)
            end
            for segment in (seg1, seg2)
                segment.length = segment_length(boundary, segment)
                boundary.length += segment.length
                push!(element.boundarySegments, length(boundary.segments) + 1)
                push!(boundary.segments, segment)
            end
            element.status = lsfSum > 0 ? ELEMENT_CENTRE_INSIDE : ELEMENT_CENTRE_OUTSIDE
        elseif nCut == 0 && element.status != ELEMENT_INSIDE
            boundary_nodes = Int[]
            for j in 1:4
                if (mesh.nodes[element.nodes[j]].status & NODE_BOUNDARY) != 0
                    push!(boundary_nodes, element.nodes[j])
                end
            end
            index, coord = is_added(boundary, boundary_nodes[1], 0, 0.0)
            if index < 0
                index = add_point!(boundary, coord, size_lambdas)
                push!(mesh.nodes[boundary_nodes[1]].boundaryPoints, index)
            end
            start = index
            index, coord = is_added(boundary, boundary_nodes[2], 0, 0.0)
            if index < 0
                index = add_point!(boundary, coord, size_lambdas)
                push!(mesh.nodes[boundary_nodes[2]].boundaryPoints, index)
            end
            segment = LevelSetBoundarySegment(start, index, i, 0.0, 0.0)
            segment.length = segment_length(boundary, segment)
            boundary.length += segment.length
            push!(element.boundarySegments, length(boundary.segments) + 1)
            push!(boundary.segments, segment)
        end
    end

    compute_point_lengths!(boundary)
    return boundary
end

function segment_length(boundary::LevelSetBoundary, segment::LevelSetBoundarySegment)
    p1 = boundary.points[segment.start].coord
    p2 = boundary.points[segment.stop].coord
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return sqrt(dx * dx + dy * dy)
end

function compute_point_lengths!(boundary::LevelSetBoundary)
    for (k, segment) in enumerate(boundary.segments)
        for idx in (segment.start, segment.stop)
            point = boundary.points[idx]
            point.length += 0.5 * segment.length
            push!(point.segments, k)
        end
        push!(boundary.points[segment.start].neighbours, segment.stop)
        push!(boundary.points[segment.stop].neighbours, segment.start)
    end
end

"""
    compute_area_fractions!(boundary)

Compute the material area fraction of every cell from the discretized
boundary and return the total area.
"""
function compute_area_fractions!(boundary::LevelSetBoundary)
    mesh = boundary.levelSet.mesh
    boundary.area = 0.0
    for element in mesh.elements
        if (element.status & ELEMENT_INSIDE) != 0
            element.area = 1.0
        elseif (element.status & ELEMENT_OUTSIDE) != 0
            element.area = 0.0
        else
            element.area = cut_area(boundary, element)
        end
        boundary.area += element.area
    end
    return boundary.area
end

function cut_area(boundary::LevelSetBoundary, element::Element)
    mesh = boundary.levelSet.mesh
    status = (element.status & ELEMENT_CENTRE_OUTSIDE) != 0 ? NODE_OUTSIDE : NODE_INSIDE
    vertices = Coord[]
    for i in 1:4
        node = element.nodes[i]
        if (mesh.nodes[node].status & status) != 0
            push!(vertices, mesh.nodes[node].coord)
        elseif (mesh.nodes[node].status & NODE_BOUNDARY) != 0
            n1 = element.nodes[i == 4 ? 1 : i + 1]
            n2 = element.nodes[i == 1 ? 4 : i - 1]
            if (mesh.nodes[n1].status & NODE_INSIDE) != 0 &&
                (mesh.nodes[n2].status & NODE_INSIDE) != 0
                push!(vertices, mesh.nodes[node].coord)
            end
        end
    end
    for segidx in element.boundarySegments
        segment = boundary.segments[segidx]
        push!(vertices, boundary.points[segment.start].coord)
        push!(vertices, boundary.points[segment.stop].coord)
    end
    area = polygon_area(vertices, element.coord)
    if (element.status & ELEMENT_CENTRE_OUTSIDE) != 0
        return 1.0 - area
    end
    return area
end

# Whether `point1` is clockwise of `point2` about `centre`.
function is_clockwise(point1::Coord, point2::Coord, centre::Coord)
    if (point1.x - centre.x) >= 0 && (point2.x - centre.x) < 0
        return false
    end
    if (point1.x - centre.x) < 0 && (point2.x - centre.x) >= 0
        return true
    end
    if (point1.x - centre.x) == 0 && (point2.x - centre.x) == 0
        if (point1.y - centre.y) >= 0 || (point2.y - centre.y) >= 0
            return point1.y > point2.y ? false : true
        end
        return point2.y > point1.y ? false : true
    end
    det =
        (point1.x - centre.x) * (point2.y - centre.y) -
        (point2.x - centre.x) * (point1.y - centre.y)
    return det >= 0
end

function polygon_area(vertices::Vector{Coord}, centre::Coord)
    n = length(vertices)
    n == 0 && return 0.0
    vs = sort(vertices; lt=(a, b) -> is_clockwise(a, b, centre))
    area = 0.0
    for i in 1:n
        j = i == n ? 1 : i + 1
        area += vs[i].x * vs[j].y
        area -= vs[j].x * vs[i].y
    end
    return abs(0.5 * area)
end
