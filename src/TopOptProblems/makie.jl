## Credit to Simon Danisch for most of the following code

const juafem_to_vtk = Dict( Triangle => 5, 
                            QuadraticTriangle => 22, 
                            Quadrilateral => 9, 
                            QuadraticQuadrilateral => 23, 
                            Tetrahedron => 10, 
                            QuadraticTetrahedron => 24, 
                            Hexahedron => 12, 
                            QuadraticHexahedron => 25
                          )

function VTKDataTypes.VTKUnstructuredData(grid::JuAFEM.Grid{dim, N}) where {dim, N}
    celltype = juafem_to_vtk[eltype(grid.cells)]
    celltypes = [celltype for i in 1:length(grid.cells)]
    connectivity = copy(reinterpret(NTuple{N, Int}, grid.cells))
    node_coords = copy(reshape(reinterpret(Float64, grid.nodes), dim, length(grid.nodes)))
    return VTKUnstructuredData(node_coords, celltypes, connectivity)
end
function VTKDataTypes.VTKUnstructuredData(problem::AbstractTopOptProblem)
    return VTKUnstructuredData(getdh(problem).grid)
end
VTKDataTypes.GLMesh(grid::JuAFEM.Grid; kwargs...) = GLMesh(VTKUnstructuredData(grid); kwargs...)
VTKDataTypes.GLMesh(problem::AbstractTopOptProblem; kwargs...) = GLMesh(VTKUnstructuredData(problem); kwargs...)

function VTKDataTypes.GLMesh(problem, topology; kwargs...)
    mesh = VTKUnstructuredData(problem)
    topology = round.(topology)
    inds = findall(isequal(0), topology)
    deleteat!(mesh.cell_connectivity, inds)
    deleteat!(mesh.cell_types, inds)
    topology = topology[setdiff(1:length(topology), inds)]
    mesh.cell_data["topology"] = topology
    return GLMesh(mesh, color = "topology")
end

#=
function AbstractPlotting.to_vertices(cells::AbstractVector{<: JuAFEM.Node{N, T}}) where {N, T}
    convert_arguments(nothing, cells)[1]
end

function AbstractPlotting.to_gl_indices(cells::AbstractVector{<: JuAFEM.Cell})
    tris = GLTriangle[]
    for cell in cells
        to_triangle(tris, cell)
    end
    tris
end

function to_triangle(tris, cell::Union{JuAFEM.Hexahedron, JuAFEM.QuadraticHexahedron})
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[5]))
    push!(tris, GLTriangle(nodes[5], nodes[2], nodes[6]))

    push!(tris, GLTriangle(nodes[6], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[6], nodes[7]))

    push!(tris, GLTriangle(nodes[7], nodes[8], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[8], nodes[4]))

    push!(tris, GLTriangle(nodes[4], nodes[8], nodes[5]))
    push!(tris, GLTriangle(nodes[5], nodes[4], nodes[1]))

    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[1], nodes[4]))
end

function to_triangle(tris, cell::Union{JuAFEM.Tetrahedron, JuAFEM.QuadraticTetrahedron})
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[3], nodes[2]))
    push!(tris, GLTriangle(nodes[3], nodes[4], nodes[2]))
    push!(tris, GLTriangle(nodes[4], nodes[3], nodes[1]))
    push!(tris, GLTriangle(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(tris, cell::Union{JuAFEM.Quadrilateral, JuAFEM.QuadraticQuadrilateral})
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::Union{JuAFEM.Triangle, JuAFEM.QuadraticTriangle})
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
end

function AbstractPlotting.convert_arguments(P, x::AbstractVector{<: JuAFEM.Node{N, T}}) where {N, T}
    convert_arguments(P, reinterpret(Point{N, T}, x))
end

function visualize(mesh::JuAFEM.Grid{dim}, u) where {dim}
    T = eltype(u)
    nnodes = length(mesh.nodes)
    #TODO make this work without creating a Node
    if dim == 2
        nodes = broadcast(1:nnodes, mesh.nodes) do i, node
            JuAFEM.Node(ntuple(Val{3}) do j
                if j < 3
                    node.x[j]
                else
                    zero(T)
                end
            end)
        end
        u = [u; zeros(T, 1, nnodes)]
    else
        nodes = mesh.nodes
    end

    cnode = AbstractPlotting.Node(zeros(Float32, length(mesh.nodes)))
    scene = Makie.mesh(nodes, mesh.cells, color = cnode, colorrange = (0.0, 33.0), shading = false);
    mplot = scene[end]
    displacevec = reinterpret(GeometryTypes.Vec{3, Float64}, u, (size(u, 2),))
    displace = norm.(displacevec)
    new_nodes = broadcast(1:length(mesh.nodes), nodes) do i, node
        JuAFEM.Node(ntuple(Val{3}) do j
            node.x[j] + u[j, i]
        end)
    end
    mesh!(nodes, mesh.cells, color = (:gray, 0.4))

    scatter!(Point3f0.(getfield.(new_nodes, :x)), markersize = 0.1);
    # TODO make mplot[1] = new_nodes work
    mplot.input_args[1][] = new_nodes
    # TODO make mplot[:color] = displace work
    push!(cnode, displace)
    points = reinterpret(Point{3, Float64}, nodes)
    #arrows!(points, displacevec, linecolor = (:black, 0.3))

    scene
end

function visualize(problem::StiffnessTopOptProblem{dim, T}, u) where {dim, T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = reshape(u[node_dofs], dim, nnodes)
    visualize(mesh, node_displacements)
end

function visualize(problem::StiffnessTopOptProblem{dim, T}) where {dim, T}
    mesh = problem.ch.dh.grid
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements)
end

function visualize(problem::StiffnessTopOptProblem, topology::AbstractVector)
    old_grid = problem.ch.dh.grid
    new_grid = JuAFEM.Grid(old_grid.cells[Bool.(round.(Int, topology))], old_grid.nodes)
    visualize(new_grid)
end

function visualize(mesh::JuAFEM.Grid{dim, N, T}) where {dim, N, T}
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements)
end
=#
