# TODO wrap Plotting into its submodule

## Credit to Simon Danisch for most of the following code
import GeometryBasics, GeometryTypes
import AbstractPlotting
import Makie
using GeometryBasics: GLTriangleFace
# using AbstractPlotting: Point2f0, Point3f0

"""
map JuAFEM cell type to VTKDataTypes cell type
"""
const juafem_to_vtk = Dict( Triangle => 5, 
                            QuadraticTriangle => 22, 
                            Quadrilateral => 9, 
                            QuadraticQuadrilateral => 23, 
                            Tetrahedron => 10, 
                            QuadraticTetrahedron => 24, 
                            Hexahedron => 12, 
                            QuadraticHexahedron => 25
                          )

"""
https://github.com/mohamed82008/VTKDataTypes.jl
"""
function VTKDataTypes.VTKUnstructuredData(grid::JuAFEM.Grid{dim, <:JuAFEM.Cell{dim,N,M}, T}) where {dim, N, M, T}
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

```
workaround taken from https://github.com/JuliaPlots/Makie.jl/issues/647
TODO: should directly convert VTKDataTypes.VTKUnstructuredData to GeometryBasics.Mesh
Do not want to spend more time on this now...
```
function GeometryBasics.Mesh(glmesh::GeometryTypes.GLNormalVertexcolorMesh)
    newverts = reinterpret(GeometryBasics.Point{3, Float32}, glmesh.vertices)
    newfaces = reinterpret(GeometryBasics.NgonFace{3, GeometryBasics.OffsetInteger{-1, UInt32}}, glmesh.faces)
    newnormals = reinterpret(GeometryBasics.Vec{3, Float32}, glmesh.normals)
    return GeometryBasics.Mesh(GeometryBasics.meta(newverts; normals = newnormals, color = glmesh.color), newfaces)
end

"""
Get mesh of the topopt problem with a given topology indicator vector
"""
function GeometryBasics.Mesh(problem::AbstractTopOptProblem, topology::Array{T,1}; kwargs...) where {T}
    mesh = VTKUnstructuredData(problem)
    topology = round.(topology)
    inds = findall(isequal(0), topology)
    deleteat!(mesh.cell_connectivity, inds)
    deleteat!(mesh.cell_types, inds)
    topology = topology[setdiff(1:length(topology), inds)]
    mesh.cell_data["topology"] = topology
    glmesh = GLMesh(mesh, color = "topology"; kwargs...)
    return GeometryBasics.Mesh(glmesh)
end

################################
# visualize problem, including supports and loads

# using AbstractPlotting.MakieLayout
# using Makie
# using TopOpt.TopOptProblems: getdim
# using TrussTopOpt.TrussTopOptProblems: TrussProblem, get_fixities_node_set_name
# using JuAFEM
# using LinearAlgebra: norm

# function visualize(problem::AbstractTopOptProblem; kwargs...)
#     scene, layout = layoutscene() #resolution = (1200, 900)
#     visualize!(scene, layout, problem; kwargs...)
#     display(scene)
#     return scene, layout
# end

# function visualize!(scene, layout, problem::AbstractTopOptProblem; u=nothing)
#     ndim = getdim(problem)
#     ncells = JuAFEM.getncells(problem)
#     nnodes = JuAFEM.getnnodes(problem)

#     color = :black

#     nodes = problem.truss_grid.grid.nodes
#     PtT = ndim == 2 ? Point2f0 : Point3f0
#     edges_pts = [PtT(nodes[cell.nodes[1]].x) => PtT(nodes[cell.nodes[2]].x) for cell in problem.truss_grid.grid.cells]

#     if ndim == 2
#         ax1 = layout[1, 1] = Axis(scene)
#         # tightlimits!(ax1)
#         # ax1.aspect = AxisAspect(1)
#         ax1.aspect = DataAspect()
#     else
#         # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
#         # https://makie.juliaplots.org/stable/cameras.html#D-Camera
#         ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
#     end
#     # TODO show the ground mesh in another Axis https://makie.juliaplots.org/stable/makielayout/grids.html
#     # ax1.title = "Truss TopOpt result"

#     # sl1 = layout[2, 1] = LSlider(scene, range = 0.01:0.01:10, startvalue = 1.0)
#     lsgrid = labelslidergrid!(scene,
#         ["support scale", "load scale", "arrow size", "vector linewidth", "element linewidth"],
#         Ref(LinRange(0.01:0.01:10)); # same range for every slider via broadcast
#         # formats = [x -> "$(round(x, digits = 2))$s" for s in ["", "", ""]],
#         # width = 350,
#         tellheight = false,
#     )
#     set_close_to!(lsgrid.sliders[1], 1.0)
#     set_close_to!(lsgrid.sliders[2], 1.0)
#     set_close_to!(lsgrid.sliders[3], 0.2)
#     set_close_to!(lsgrid.sliders[4], 6.0)
#     set_close_to!(lsgrid.sliders[5], 6.0)
#     arrow_size = lift(s -> s, lsgrid.sliders[3].value)
#     arrow_linewidth = lift(s -> s, lsgrid.sliders[4].value)
#     layout[2, 1] = lsgrid.layout

#     # * draw element
#     # http://juliaplots.org/MakieReferenceImages/gallery//tutorial_linesegments/index.html
#     element_linewidth = lift(s -> a.*s, lsgrid.sliders[5].value)
#     linesegments!(ax1, edges_pts, 
#                   linewidth = element_linewidth,
#                   color = color)

#     # * draw displacements
#     if u !== nothing
#         node_dofs = problem.metadata.node_dofs
#         @assert length(u) == ndim * nnodes

#         exagg_ls = labelslider!(scene, "deformation exaggeration:", 0:0.01:1000.0)
#         set_close_to!(exagg_ls.slider, 1.0)
#         exagg_edge_pts = lift(s -> [PtT(nodes[cell.nodes[1]].x) + PtT(u[node_dofs[:,cell.nodes[1]]]*s) => PtT(nodes[cell.nodes[2]].x) + PtT(u[node_dofs[:,cell.nodes[2]]]*s) for cell in problem.truss_grid.grid.cells], exagg_ls.slider.value)
#         layout[3, 1] = exagg_ls.layout

#         linesegments!(ax1, exagg_edge_pts, 
#                       linewidth = element_linewidth,
#                       color = :cyan)
#     end

#     # fixties vectors
#     for i=1:ndim
#         nodeset_name = get_fixities_node_set_name(i)
#         fixed_node_ids = JuAFEM.getnodeset(problem.truss_grid.grid, nodeset_name)
#         dir = zeros(ndim)
#         dir[i] = 1.0
#         scaled_base_pts = lift(s->[PtT(nodes[node_id].x) - PtT(dir*s) for node_id in fixed_node_ids], 
#             lsgrid.sliders[1].value)
#         scaled_fix_dirs = lift(s->fill(PtT(dir*s), length(fixed_node_ids)), lsgrid.sliders[1].value)
#         Makie.arrows!(
#             ax1,
#             scaled_base_pts,
#             scaled_fix_dirs,
#             arrowcolor=:orange,
#             arrowsize=arrow_size,
#             linecolor=:orange,
#             linewidth=arrow_linewidth,
#         )
#     end
#     # load vectors
#     scaled_load_dirs = lift(s->[PtT(force/norm(force)*s) for force in values(problem.force)], 
#         lsgrid.sliders[2].value)
#     Makie.arrows!(
#         ax1,
#         [PtT(nodes[node_id].x) for node_id in keys(problem.force)],
#         scaled_load_dirs,
#         arrowcolor=:purple,
#         arrowsize=arrow_size,
#         linecolor=:purple,
#         linewidth=arrow_linewidth,
#     )
# end


################################
#  archived below

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/f16321dee2c77ac9c753fed9b1074a2df7b10db8/src/utilities/utilities.jl#L188
# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L522
function AbstractPlotting.to_vertices(nodes::Vector{<:JuAFEM.Node})
    return AbstractPlotting.Point3f0.([n.x for n in nodes])
end

function AbstractPlotting.to_triangles(cells::AbstractVector{<: JuAFEM.Cell})
    tris = GLTriangleFace[]
    for cell in cells
        to_triangle(tris, cell)
    end
    tris
end

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L505
function to_triangle(tris, cell::Union{JuAFEM.Hexahedron, JuAFEM.QuadraticHexahedron})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[5]))
    push!(tris, GLTriangleFace(nodes[5], nodes[2], nodes[6]))

    push!(tris, GLTriangleFace(nodes[6], nodes[2], nodes[3]))
    push!(tris, GLTriangleFace(nodes[3], nodes[6], nodes[7]))

    push!(tris, GLTriangleFace(nodes[7], nodes[8], nodes[3]))
    push!(tris, GLTriangleFace(nodes[3], nodes[8], nodes[4]))

    push!(tris, GLTriangleFace(nodes[4], nodes[8], nodes[5]))
    push!(tris, GLTriangleFace(nodes[5], nodes[4], nodes[1]))

    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangleFace(nodes[3], nodes[1], nodes[4]))
end

function to_triangle(tris, cell::Union{JuAFEM.Tetrahedron, JuAFEM.QuadraticTetrahedron})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[3], nodes[2]))
    push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[2]))
    push!(tris, GLTriangleFace(nodes[4], nodes[3], nodes[1]))
    push!(tris, GLTriangleFace(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(tris, cell::Union{JuAFEM.Quadrilateral, JuAFEM.QuadraticQuadrilateral})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::Union{JuAFEM.Triangle, JuAFEM.QuadraticTriangle})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
end

function AbstractPlotting.convert_arguments(P, x::AbstractVector{<:JuAFEM.Node{N, T}}) where {N, T}
    convert_arguments(P, reinterpret(Point{N, T}, x))
end

function visualize(mesh::JuAFEM.Grid{dim, <:JuAFEM.AbstractCell, TT}, u) where {dim, TT}
    T = eltype(u)
    nnodes = length(mesh.nodes)
    #TODO make this work without creating a Node
    if dim == 2
        nodes = Vector{JuAFEM.Node}(undef, nnodes)
        for (i, node) in enumerate(mesh.nodes)
            nodes[i] = JuAFEM.Node((node.x[1], node.x[2], zero(T)))
        end
        # nodes = broadcast(1:nnodes, mesh.nodes) do i, node
        #     JuAFEM.Node(ntuple(Val{3}) do j
        #         if j < 3
        #             node.x[j]
        #         else
        #             zero(T)
        #         end
        #     end)
        # end
        u = [u; zeros(T, 1, nnodes)]
    else
        nodes = mesh.nodes
    end

    cnode = AbstractPlotting.Node(zeros(Float32, length(mesh.nodes)))
    scene = Makie.mesh(nodes, mesh.cells, color = cnode, colorrange = (0.0, 33.0), shading = false);

    # mplot = scene[end]

    # new_nodes = broadcast(1:length(mesh.nodes), nodes) do i, node
    #     JuAFEM.Node(ntuple(Val{3}) do j
    #         node.x[j] + u[j, i]
    #     end)
    # end
    new_nodes = Vector{JuAFEM.Node}(undef, length(nodes))
    for (i, node) in enumerate(nodes)
        new_nodes[i] = JuAFEM.Node(Tuple([node.x[j] + u[j, i] for j=1:3]))
    end
    Makie.mesh!(scene, new_nodes, mesh.cells, color = (:gray, 0.4))

    Makie.scatter!(AbstractPlotting.Point3f0.(getfield.(new_nodes, :x)), markersize = 0.1);
    # # TODO make mplot[1] = new_nodes work
    # mplot.input_args[1][] = new_nodes
    # # TODO make mplot[:color] = displace work
    # push!(cnode, displace)

    points = AbstractPlotting.Point3f0.(getfield.(nodes, :x))
    # GeometryTypes.Vec{3, Float64}
    # displacevec = reinterpret(AbstractPlotting.Vec3f0, u, (size(u, 2),))
    displacevec = AbstractPlotting.Vec3f0.([u[:,i] for i=1:size(u,2)])
    displace = norm.(displacevec)
    Makie.arrows!(points, displacevec, linecolor = (:black, 0.3))

    scene
end

"""
draw problem's initial grid with a given displacement vector `u`
"""
function visualize(problem::StiffnessTopOptProblem{dim, T}, u) where {dim, T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = reshape(u[node_dofs], dim, nnodes)
    visualize(mesh, node_displacements)
end

"""
draw initial grid
"""
function visualize(problem::StiffnessTopOptProblem{dim, T}) where {dim, T}
    mesh = problem.ch.dh.grid
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements)
end

# function visualize(problem::StiffnessTopOptProblem, topology::AbstractVector)
#     old_grid = problem.ch.dh.grid
#     new_grid = JuAFEM.Grid(old_grid.cells[Bool.(round.(Int, topology))], old_grid.nodes)
#     visualize(new_grid)
# end

function visualize(mesh::JuAFEM.Grid{dim, N, T}) where {dim, N, T}
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements)
end