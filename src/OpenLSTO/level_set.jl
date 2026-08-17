# Level-set function on a fixed grid (a port of `M2DO_LSM/src/level_set.cpp`).
#
# The signed distance is positive inside the structure and negative outside.
# The zero contour is advanced by a normal velocity extended from the boundary
# points through the narrow band, with the spatial gradient approximated by a
# 5th-order Hamilton-Jacobi WENO upwind stencil.

mutable struct LevelSet
    mesh::Mesh
    moveLimit::Float64
    bandWidth::Int
    isFixed::Bool
    signedDistance::Vector{Float64}
    velocity::Vector{Float64}
    gradient::Vector{Float64}
    narrowBand::Vector{Int}
    mines::Vector{Int}
end

function LevelSet(
    mesh::Mesh, moveLimit::Real=0.5, bandWidth::Integer=6, isFixed::Bool=false
)
    bandWidth > 2 || error("Width of the narrow band must be greater than 2.")
    0 < moveLimit <= 1 || error("Move limit must be between 0 and 1.")
    ls = LevelSet(
        mesh,
        Float64(moveLimit),
        Int(bandWidth),
        isFixed,
        zeros(mesh.nNodes),
        zeros(mesh.nNodes),
        zeros(mesh.nNodes),
        Int[],
        Int[],
    )
    initialise!(ls, swiss_cheese_holes(mesh.width, mesh.height))
    initialise_narrow_band!(ls)
    return ls
end

function LevelSet(
    mesh::Mesh,
    holes::Vector{Hole},
    moveLimit::Real=0.5,
    bandWidth::Integer=6,
    isFixed::Bool=false,
)
    bandWidth > 2 || error("Width of the narrow band must be greater than 2.")
    0 < moveLimit <= 1 || error("Move limit must be between 0 and 1.")
    ls = LevelSet(
        mesh,
        Float64(moveLimit),
        Int(bandWidth),
        isFixed,
        zeros(mesh.nNodes),
        zeros(mesh.nNodes),
        zeros(mesh.nNodes),
        Int[],
        Int[],
    )
    initialise!(ls, holes)
    initialise_narrow_band!(ls)
    return ls
end

# Signed distance to the closest domain boundary.
function closest_domain_boundary!(ls::LevelSet)
    mesh = ls.mesh
    for i in eachindex(mesh.nodes)
        coord = mesh.nodes[i].coord
        min_x = min(coord.x, mesh.width - coord.x)
        min_y = min(coord.y, mesh.height - coord.y)
        ls.signedDistance[i] = min(min_x, min_y)
    end
end

# Initialise the signed distance from a set of circular holes.
function initialise!(ls::LevelSet, holes::Vector{Hole})
    mesh = ls.mesh
    closest_domain_boundary!(ls)
    for i in eachindex(mesh.nodes)
        coord = mesh.nodes[i].coord
        for hole in holes
            dx = hole.coord.x - coord.x
            dy = hole.coord.y - coord.y
            dist = sqrt(dx * dx + dy * dy) - hole.r
            if dist < ls.signedDistance[i]
                ls.signedDistance[i] = dist
            end
        end
    end
end

# Distance from a point to the closest domain boundary (helper for
# `initialise!(ls, points)` below).
function point_to_line_distance(v1::Coord, v2::Coord, p::Coord)
    dx = v2.x - v1.x
    dy = v2.y - v1.y
    rSqd = dx * dx + dy * dy
    if rSqd < 1e-6
        dx = p.x - v1.x
        dy = p.y - v1.y
        return sqrt(dx * dx + dy * dy)
    end
    t = ((p.x - v1.x) * dx + (p.y - v1.y) * dy) / rSqd
    t = clamp(t, 0.0, 1.0)
    x = v1.x + t * dx
    y = v1.y + t * dy
    dx = x - p.x
    dy = y - p.y
    return sqrt(dx * dx + dy * dy)
end

function is_left_of_line(v1::Coord, v2::Coord, p::Coord)
    return (v2.x - v1.x) * (p.y - v1.y) - (p.x - v1.x) * (v2.y - v1.y)
end

# Winding-number point-in-polygon test (vertices closed and ordered).
function is_inside_polygon(p::Coord, vertices::Vector{Coord})
    windingNumber = 0
    for i in 1:(length(vertices) - 1)
        if vertices[i].y <= p.y
            if vertices[i + 1].y > p.y
                if is_left_of_line(vertices[i], vertices[i + 1], p) > 0
                    windingNumber += 1
                end
            end
        else
            if vertices[i + 1].y <= p.y
                if is_left_of_line(vertices[i], vertices[i + 1], p) < 0
                    windingNumber -= 1
                end
            end
        end
    end
    return windingNumber != 0
end

# Initialise the signed distance from a closed piecewise-linear interface.
function initialise!(ls::LevelSet, points::Vector{Coord})
    mesh = ls.mesh
    closest_domain_boundary!(ls)
    for i in eachindex(mesh.nodes)
        coord = mesh.nodes[i].coord
        for j in 1:(length(points) - 1)
            dist = point_to_line_distance(points[j], points[j + 1], coord)
            if dist < ls.signedDistance[i]
                ls.signedDistance[i] = dist
            end
        end
        if is_inside_polygon(coord, points)
            ls.signedDistance[i] *= -1
        end
    end
end

function initialise_narrow_band!(ls::LevelSet)
    mesh = ls.mesh
    mineWidth = ls.bandWidth - 1
    empty!(ls.narrowBand)
    empty!(ls.mines)
    for i in eachindex(mesh.nodes)
        node = mesh.nodes[i]
        node.isActive = false
        node.isMine = false
        if !node.isFixed && (!node.isDomain || !ls.isFixed)
            asd = abs(ls.signedDistance[i])
            if asd < ls.bandWidth
                node.isActive = true
                push!(ls.narrowBand, i)
                if asd > mineWidth
                    node.isMine = true
                    push!(ls.mines, i)
                end
            end
        end
    end
end

# Map boundary-point velocities to level-set nodes using inverse-squared
# distance interpolation.
function initialise_velocities!(ls::LevelSet, boundaryPoints::Vector{BoundaryPoint})
    mesh = ls.mesh
    isSet = falses(mesh.nNodes)
    weight = zeros(mesh.nNodes)
    fill!(ls.velocity, 0.0)
    for bp in boundaryPoints
        node = get_closest_node(mesh, bp.coord)
        dx = mesh.nodes[node].coord.x - bp.coord.x
        dy = mesh.nodes[node].coord.y - bp.coord.y
        rSqd = dx * dx + dy * dy
        if rSqd < 1e-6
            ls.velocity[node] = bp.velocity
            weight[node] = 1.0
            isSet[node] = true
        elseif !isSet[node]
            ls.velocity[node] += bp.velocity / rSqd
            weight[node] += 1.0 / rSqd
        end
        for j in 1:4
            neighbour = mesh.nodes[node].neighbours[j]
            if neighbour != 0
                dx = mesh.nodes[neighbour].coord.x - bp.coord.x
                dy = mesh.nodes[neighbour].coord.y - bp.coord.y
                rSqd = dx * dx + dy * dy
                if rSqd < 1e-6
                    ls.velocity[neighbour] = bp.velocity
                    weight[neighbour] = 1.0
                    isSet[neighbour] = true
                elseif rSqd <= 1.0 && !isSet[neighbour]
                    ls.velocity[neighbour] += bp.velocity / rSqd
                    weight[neighbour] += 1.0 / rSqd
                end
            end
        end
    end
    for node in ls.narrowBand
        if ls.velocity[node] != 0
            ls.velocity[node] /= weight[node]
        end
    end
end

function reinitialise!(ls::LevelSet)
    fmm = FastMarchingMethod(ls.mesh)
    march!(fmm, ls.signedDistance)
    initialise_narrow_band!(ls)
    return nothing
end

# Extend boundary-point velocities to every narrow-band node.
function compute_velocities!(ls::LevelSet, boundaryPoints::Vector{BoundaryPoint})
    initialise_velocities!(ls, boundaryPoints)
    fmm = FastMarchingMethod(ls.mesh)
    march!(fmm, ls.signedDistance, ls.velocity)
    return nothing
end

function compute_gradients!(ls::LevelSet)
    fill!(ls.gradient, 0.0)
    for node in ls.narrowBand
        ls.gradient[node] = compute_gradient(ls, node)
    end
end

# Advance the zero contour and reinitialise when the front nears the narrow
# band edge. Returns whether a reinitialisation was performed.
function update!(ls::LevelSet, timeStep::Float64)
    mesh = ls.mesh
    for node in ls.narrowBand
        ls.signedDistance[node] -= timeStep * ls.gradient[node] * ls.velocity[node]
        if mesh.nodes[node].isDomain && ls.signedDistance[node] > 0
            ls.signedDistance[node] = 0.0
        end
        empty!(mesh.nodes[node].boundaryPoints)
    end
    for mine in ls.mines
        if abs(ls.signedDistance[mine]) < 1.0
            reinitialise!(ls)
            return true
        end
    end
    return false
end

function signed_distance_at(ls::LevelSet, x::Int, y::Int)
    return ls.signedDistance[xy_to_index(ls.mesh, x, y)]
end

# Modulus of the gradient of the signed distance at a node, using the
# Hamilton-Jacobi WENO upwind approximation (see `LevelSet::computeGradient`).
function compute_gradient(ls::LevelSet, node::Int)
    mesh = ls.mesh
    x = node_x(mesh, node)
    y = node_y(mesh, node)
    lsf = ls.signedDistance[node]
    isGradient = false
    grad = 0.0

    # Corner cases where the zero contour runs diagonally through the node.
    if x == 0
        if y == 0
            if abs(signed_distance_at(ls, x + 1, y) - lsf) < 1e-6 &&
                abs(signed_distance_at(ls, x, y + 1) - lsf) < 1e-6
                grad = abs(lsf - signed_distance_at(ls, x + 1, y + 1)) * sqrt(2.0)
                isGradient = true
            end
        elseif y == mesh.height
            if abs(signed_distance_at(ls, x + 1, y) - lsf) < 1e-6 &&
                abs(signed_distance_at(ls, x, y - 1) - lsf) < 1e-6
                grad = abs(lsf - signed_distance_at(ls, x + 1, y - 1)) * sqrt(2.0)
                isGradient = true
            end
        end
    elseif x == mesh.width
        if y == 0
            if abs(signed_distance_at(ls, x - 1, y) - lsf) < 1e-6 &&
                abs(signed_distance_at(ls, x, y + 1) - lsf) < 1e-6
                grad = abs(lsf - signed_distance_at(ls, x - 1, y + 1)) * sqrt(2.0)
                isGradient = true
            end
        elseif y == mesh.height
            if abs(signed_distance_at(ls, x - 1, y) - lsf) < 1e-6 &&
                abs(signed_distance_at(ls, x, y - 1) - lsf) < 1e-6
                grad = abs(lsf - signed_distance_at(ls, x - 1, y - 1)) * sqrt(2.0)
                isGradient = true
            end
        end
    end

    if !isGradient
        v1 = v2 = v3 = v4 = v5 = 0.0
        sign = ls.velocity[node] < 0 ? -1.0 : 1.0

        # Derivatives to the right.
        if x == 0
            v1 = signed_distance_at(ls, 3, y) - signed_distance_at(ls, 2, y)
            v2 = signed_distance_at(ls, 2, y) - signed_distance_at(ls, 1, y)
            v3 = signed_distance_at(ls, 1, y) - signed_distance_at(ls, 0, y)
            v4 = v3
            v5 = v3
        elseif x == 1
            v1 = signed_distance_at(ls, 4, y) - signed_distance_at(ls, 3, y)
            v2 = signed_distance_at(ls, 3, y) - signed_distance_at(ls, 2, y)
            v3 = signed_distance_at(ls, 2, y) - signed_distance_at(ls, 1, y)
            v4 = signed_distance_at(ls, 1, y) - signed_distance_at(ls, 0, y)
            v5 = v4
        elseif x == mesh.width
            v5 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v3 = v4
            v2 = v4
            v1 = v4
        elseif x == mesh.width - 1
            v5 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v3 = signed_distance_at(ls, x + 1, y) - signed_distance_at(ls, x, y)
            v2 = v3
            v1 = v3
        elseif x == mesh.width - 2
            v5 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v3 = signed_distance_at(ls, x + 1, y) - signed_distance_at(ls, x, y)
            v2 = signed_distance_at(ls, x + 2, y) - signed_distance_at(ls, x + 1, y)
            v1 = v2
        else
            v1 = signed_distance_at(ls, x + 3, y) - signed_distance_at(ls, x + 2, y)
            v2 = signed_distance_at(ls, x + 2, y) - signed_distance_at(ls, x + 1, y)
            v3 = signed_distance_at(ls, x + 1, y) - signed_distance_at(ls, x, y)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v5 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
        end
        gradRight = sign * grad_HJ_WENO(v1, v2, v3, v4, v5)

        # Derivatives to the left.
        if x == mesh.width
            v1 = signed_distance_at(ls, x - 2, y) - signed_distance_at(ls, x - 3, y)
            v2 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v4 = v3
            v5 = v3
        elseif x == mesh.width - 1
            v1 = signed_distance_at(ls, x - 2, y) - signed_distance_at(ls, x - 3, y)
            v2 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v4 = signed_distance_at(ls, x + 1, y) - signed_distance_at(ls, x, y)
            v5 = v4
        elseif x == 0
            v5 = signed_distance_at(ls, 2, y) - signed_distance_at(ls, 1, y)
            v4 = signed_distance_at(ls, 1, y) - signed_distance_at(ls, 0, y)
            v3 = v4
            v2 = v4
            v1 = v4
        elseif x == 1
            v5 = signed_distance_at(ls, 3, y) - signed_distance_at(ls, 2, y)
            v4 = signed_distance_at(ls, 2, y) - signed_distance_at(ls, 1, y)
            v3 = signed_distance_at(ls, 1, y) - signed_distance_at(ls, 0, y)
            v2 = v3
            v1 = v3
        elseif x == 2
            v5 = signed_distance_at(ls, 4, y) - signed_distance_at(ls, 3, y)
            v4 = signed_distance_at(ls, 3, y) - signed_distance_at(ls, 2, y)
            v3 = signed_distance_at(ls, 2, y) - signed_distance_at(ls, 1, y)
            v2 = signed_distance_at(ls, 1, y) - signed_distance_at(ls, 0, y)
            v1 = v2
        else
            v1 = signed_distance_at(ls, x - 2, y) - signed_distance_at(ls, x - 3, y)
            v2 = signed_distance_at(ls, x - 1, y) - signed_distance_at(ls, x - 2, y)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x - 1, y)
            v4 = signed_distance_at(ls, x + 1, y) - signed_distance_at(ls, x, y)
            v5 = signed_distance_at(ls, x + 2, y) - signed_distance_at(ls, x + 1, y)
        end
        gradLeft = sign * grad_HJ_WENO(v1, v2, v3, v4, v5)

        # Upward derivatives.
        if y == 0
            v1 = signed_distance_at(ls, x, 3) - signed_distance_at(ls, x, 2)
            v2 = signed_distance_at(ls, x, 2) - signed_distance_at(ls, x, 1)
            v3 = signed_distance_at(ls, x, 1) - signed_distance_at(ls, x, 0)
            v4 = v3
            v5 = v3
        elseif y == 1
            v1 = signed_distance_at(ls, x, 4) - signed_distance_at(ls, x, 3)
            v2 = signed_distance_at(ls, x, 3) - signed_distance_at(ls, x, 2)
            v3 = signed_distance_at(ls, x, 2) - signed_distance_at(ls, x, 1)
            v4 = signed_distance_at(ls, x, 1) - signed_distance_at(ls, x, 0)
            v5 = v4
        elseif y == mesh.height
            v5 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v3 = v4
            v2 = v4
            v1 = v4
        elseif y == mesh.height - 1
            v5 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v3 = signed_distance_at(ls, x, y + 1) - signed_distance_at(ls, x, y)
            v2 = v3
            v1 = v3
        elseif y == mesh.height - 2
            v5 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v3 = signed_distance_at(ls, x, y + 1) - signed_distance_at(ls, x, y)
            v2 = signed_distance_at(ls, x, y + 2) - signed_distance_at(ls, x, y + 1)
            v1 = v2
        else
            v1 = signed_distance_at(ls, x, y + 3) - signed_distance_at(ls, x, y + 2)
            v2 = signed_distance_at(ls, x, y + 2) - signed_distance_at(ls, x, y + 1)
            v3 = signed_distance_at(ls, x, y + 1) - signed_distance_at(ls, x, y)
            v4 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v5 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
        end
        gradUp = sign * grad_HJ_WENO(v1, v2, v3, v4, v5)

        # Downward derivatives.
        if y == mesh.height
            v1 = signed_distance_at(ls, x, y - 2) - signed_distance_at(ls, x, y - 3)
            v2 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v4 = v3
            v5 = v3
        elseif y == mesh.height - 1
            v1 = signed_distance_at(ls, x, y - 2) - signed_distance_at(ls, x, y - 3)
            v2 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v4 = signed_distance_at(ls, x, y + 1) - signed_distance_at(ls, x, y)
            v5 = v4
        elseif y == 0
            v5 = signed_distance_at(ls, x, 2) - signed_distance_at(ls, x, 1)
            v4 = signed_distance_at(ls, x, 1) - signed_distance_at(ls, x, 0)
            v3 = v4
            v2 = v4
            v1 = v4
        elseif y == 1
            v5 = signed_distance_at(ls, x, 3) - signed_distance_at(ls, x, 2)
            v4 = signed_distance_at(ls, x, 2) - signed_distance_at(ls, x, 1)
            v3 = signed_distance_at(ls, x, 1) - signed_distance_at(ls, x, 0)
            v2 = v3
            v1 = v3
        elseif y == 2
            v5 = signed_distance_at(ls, x, 4) - signed_distance_at(ls, x, 3)
            v4 = signed_distance_at(ls, x, 3) - signed_distance_at(ls, x, 2)
            v3 = signed_distance_at(ls, x, 2) - signed_distance_at(ls, x, 1)
            v2 = signed_distance_at(ls, x, 1) - signed_distance_at(ls, x, 0)
            v1 = v2
        else
            v1 = signed_distance_at(ls, x, y - 2) - signed_distance_at(ls, x, y - 3)
            v2 = signed_distance_at(ls, x, y - 1) - signed_distance_at(ls, x, y - 2)
            v3 = signed_distance_at(ls, x, y) - signed_distance_at(ls, x, y - 1)
            v4 = signed_distance_at(ls, x, y + 1) - signed_distance_at(ls, x, y)
            v5 = signed_distance_at(ls, x, y + 2) - signed_distance_at(ls, x, y + 1)
        end
        gradDown = sign * grad_HJ_WENO(v1, v2, v3, v4, v5)

        # Upwind combination of the one-sided derivatives.
        gradDown > 0 && (grad += gradDown * gradDown)
        gradLeft > 0 && (grad += gradLeft * gradLeft)
        gradUp < 0 && (grad += gradUp * gradUp)
        gradRight < 0 && (grad += gradRight * gradRight)
        grad = sqrt(grad)
    end

    return grad
end

# 5th-order Hamilton-Jacobi WENO gradient (LevelSet::gradHJWENO).
function grad_HJ_WENO(v1::Float64, v2::Float64, v3::Float64, v4::Float64, v5::Float64)
    oneQuarter = 1.0 / 4.0
    thirteenTwelths = 13.0 / 12.0
    epsv = 1e-6

    s1 = thirteenTwelths * (v1 - 2 * v2 + v3)^2 + oneQuarter * (v1 - 4 * v2 + 3 * v3)^2
    s2 = thirteenTwelths * (v2 - 2 * v3 + v4)^2 + oneQuarter * (v2 - v4)^2
    s3 = thirteenTwelths * (v3 - 2 * v4 + v5)^2 + oneQuarter * (3 * v3 - 4 * v4 + v5)^2

    alpha1 = 0.1 / ((s1 + epsv) * (s1 + epsv))
    alpha2 = 0.6 / ((s2 + epsv) * (s2 + epsv))
    alpha3 = 0.3 / ((s3 + epsv) * (s3 + epsv))

    totalWeight = alpha1 + alpha2 + alpha3
    w1 = alpha1 / totalWeight
    w2 = alpha2 / totalWeight
    w3 = alpha3 / totalWeight

    grad =
        w1 * (2 * v1 - 7 * v2 + 11 * v3) +
        w2 * (5 * v3 - v2 + 2 * v4) +
        w3 * (2 * v3 + 5 * v4 - v5)
    return grad / 6.0
end
