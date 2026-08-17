# Level-set fixed-grid mesh (a port of `M2DO_LSM/include/mesh.h`).
#
# The grid is two-dimensional, made of unit-square elements, with one node per
# grid point. Node coordinates are `(x, y)` with `x ∈ 0:width` and
# `y ∈ 0:height`. Node indices are 1-based; the out-of-bounds neighbour
# sentinel is `0` (OpenLSTO uses `nNodes` for the same purpose).

mutable struct Node
    coord::Coord
    neighbours::Vector{Int}      # left, right, down, up
    elements::Vector{Int}        # elements the node is connected to
    boundaryPoints::Vector{Int}  # boundary points associated with the node
    isActive::Bool
    isDomain::Bool
    isMasked::Bool
    isFixed::Bool
    isMine::Bool
    status::Int                  # NODE_* flag
end

mutable struct Element
    coord::Coord                 # element centre
    area::Float64                # material area fraction
    nodes::Vector{Int}           # four node indices (bl, br, tr, tl)
    boundarySegments::Vector{Int}
    status::Int                  # ELEMENT_* flag
end

mutable struct Mesh
    width::Int
    height::Int
    nElements::Int
    nNodes::Int
    nodes::Vector{Node}
    elements::Vector{Element}

    function Mesh(width::Integer, height::Integer)
        w = Int(width)
        h = Int(height)
        nodes = [
            Node(
                Coord(0.0, 0.0),
                zeros(Int, 4),
                Int[],
                Int[],
                false,
                false,
                false,
                false,
                false,
                NODE_NONE,
            ) for _ in 1:((w + 1) * (h + 1))
        ]
        elements = [
            Element(Coord(0.0, 0.0), 0.0, zeros(Int, 4), Int[], ELEMENT_NONE) for
            _ in 1:(w * h)
        ]
        mesh = new(w, h, w * h, (w + 1) * (h + 1), nodes, elements)
        initialise_nodes!(mesh)
        initialise_elements!(mesh)
        return mesh
    end
end

# 0-based (x, y) grid coordinate -> 1-based node index.
xy_to_index(mesh::Mesh, x::Int, y::Int) = y * (mesh.width + 1) + x + 1

# 1-based node index -> 0-based grid coordinate.
node_x(mesh::Mesh, node::Int) = (node - 1) % (mesh.width + 1)
node_y(mesh::Mesh, node::Int) = (node - 1) ÷ (mesh.width + 1)

function initialise_nodes!(mesh::Mesh)
    for i in eachindex(mesh.nodes)
        node = mesh.nodes[i]
        node.isDomain = false
        node.isMasked = false
        node.isFixed = false
        node.isActive = false
        node.isMine = false
        node.status = NODE_NONE
        empty!(node.elements)
        empty!(node.boundaryPoints)

        x = (i - 1) % (mesh.width + 1)
        y = (i - 1) ÷ (mesh.width + 1)
        if x == 0 || x == mesh.width || y == 0 || y == mesh.height
            node.isDomain = true
        end
        node.coord = Coord(Float64(x), Float64(y))

        node.neighbours[1] = x > 0 ? xy_to_index(mesh, x - 1, y) : 0
        node.neighbours[2] = x < mesh.width ? xy_to_index(mesh, x + 1, y) : 0
        node.neighbours[3] = y > 0 ? xy_to_index(mesh, x, y - 1) : 0
        node.neighbours[4] = y < mesh.height ? xy_to_index(mesh, x, y + 1) : 0
    end
end

function initialise_elements!(mesh::Mesh)
    w = mesh.width + 1
    for i in 1:(mesh.nElements)
        element = mesh.elements[i]
        x = (i - 1) % mesh.width
        y = (i - 1) ÷ mesh.width

        element.coord = Coord(Float64(x) + 0.5, Float64(y) + 0.5)
        element.area = 0.0
        element.status = ELEMENT_NONE

        element.nodes[1] = x + y * w + 1          # bottom left
        element.nodes[2] = x + 1 + y * w + 1      # bottom right
        element.nodes[3] = x + 1 + (y + 1) * w + 1  # top right
        element.nodes[4] = x + (y + 1) * w + 1    # top left

        for j in 1:4
            push!(mesh.nodes[element.nodes[j]].elements, i)
        end
    end
end

function get_element(mesh::Mesh, x::Float64, y::Float64)
    x -= 1e-6
    y -= 1e-6
    x = max(0.0, x)
    y = max(0.0, y)
    element_x = floor(Int, x)
    element_y = floor(Int, y)
    return element_y * mesh.width + element_x + 1
end
get_element(mesh::Mesh, point::Coord) = get_element(mesh, point.x, point.y)

function get_closest_node(mesh::Mesh, x::Float64, y::Float64)
    element = mesh.elements[get_element(mesh, x, y)]
    dx = x - element.coord.x
    dy = y - element.coord.y
    if dx < 0
        return dy < 0 ? element.nodes[1] : element.nodes[4]
    else
        return dy < 0 ? element.nodes[2] : element.nodes[3]
    end
end
get_closest_node(mesh::Mesh, point::Coord) = get_closest_node(mesh, point.x, point.y)
