import AbstractPlotting
import Makie
using AbstractPlotting: lift, cam3d!
using AbstractPlotting.MakieLayout: layoutscene, DataAspect, Axis, labelslidergrid!, set_close_to!,
    labelslider!, LScene
using GeometryBasics: GLTriangleFace

################################
# Credit to Simon Danisch for the conversion code below

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

################################

function visualize(mesh::JuAFEM.Grid{dim, <:JuAFEM.AbstractCell, TT}, u; 
        default_support_scale=1.0, default_load_scale=1.0, scale_range=1.0, default_exagg_scale=1.0, exagg_range=1.0) where {dim, TT}
    T = eltype(u)
    nnodes = length(mesh.nodes)
    # * initialize the makie scene
    scene, layout = layoutscene() #resolution = (1200, 900)

    #TODO make this work without creating a Node
    if dim == 2
        nodes = Vector{JuAFEM.Node}(undef, nnodes)
        for (i, node) in enumerate(mesh.nodes)
            nodes[i] = JuAFEM.Node((node.x[1], node.x[2], zero(T)))
        end
        u = [u; zeros(T, 1, nnodes)]

        ax1 = layout[1, 1] = Axis(scene)
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        nodes = mesh.nodes

        # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
        # https://makie.juliaplots.org/stable/cameras.html#D-Camera
        ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
    end

    # TODO show the ground mesh in another Axis https://makie.juliaplots.org/stable/makielayout/grids.html
    # ax1.title = "TopOpt result"

    # * support / load appearance / deformatione exaggeration control
    lsgrid = labelslidergrid!(scene,
        ["support scale", "load scale", "deformation exaggeration"],
        [LinRange(0.0:0.01:scale_range), LinRange(0.0:0.01:scale_range), LinRange(0.0:0.01:exagg_range)];
        # formats = [x -> "$(round(x, digits = 2))$s" for s in ["", "", ""]],
        # width = 200,
        tellheight = false,
    )
    set_close_to!(lsgrid.sliders[1], default_support_scale)
    set_close_to!(lsgrid.sliders[2], default_load_scale)
    set_close_to!(lsgrid.sliders[3], default_exagg_scale)
    layout[2, 1] = lsgrid.layout

    # undeformed mesh
    # cnode = AbstractPlotting.Node(zeros(Float32, length(mesh.nodes)))
    Makie.mesh!(ax1, nodes, mesh.cells, color = (:gray, 0.4), shading = false);

    # * deformed mesh
    exagg_deformed_nodes = lift(s -> 
        [JuAFEM.Node(Tuple([node.x[j] + s * u[j, i] for j=1:3])) for (i, node) in enumerate(nodes)], 
        lsgrid.sliders[3].value)
    new_nodes = Vector{JuAFEM.Node}(undef, length(nodes))
    # for (i, node) in enumerate(nodes)
    #     new_nodes[i] = JuAFEM.Node(Tuple([node.x[j] + u[j, i] for j=1:3]))
    # end
    Makie.mesh!(ax1, exagg_deformed_nodes, mesh.cells, color = (:purple, 0.4))

    # Makie.scatter!(ax1, AbstractPlotting.Point3f0.(getfield.(new_nodes, :x)), markersize = 0.1);

    # points = AbstractPlotting.Point3f0.(getfield.(nodes, :x))
    # GeometryTypes.Vec{3, Float64}
    # displacevec = reinterpret(AbstractPlotting.Vec3f0, u, (size(u, 2),))
    # displacevec = AbstractPlotting.Vec3f0.([u[:,i] for i=1:size(u,2)])
    # displace = norm.(displacevec)
    # Makie.arrows!(points, displacevec, linecolor = (:black, 0.3))

    scene, layout
end

"""
draw problem's initial grid with a given displacement vector `u`
"""
function visualize(problem::StiffnessTopOptProblem{dim, T}, u; kwargs...) where {dim, T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = reshape(u[node_dofs], dim, nnodes)
    visualize(mesh, node_displacements; kwargs...)
end

"""
draw initial grid
"""
function visualize(problem::StiffnessTopOptProblem{dim, T}; kwargs...) where {dim, T}
    mesh = problem.ch.dh.grid
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements; kwargs...)
end

function visualize(mesh::JuAFEM.Grid{dim, N, T}; kwargs...) where {dim, N, T}
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    visualize(mesh, node_displacements; kwargs...)
end

# TODO visualize a given topology, with deformation vector options
# function visualize(problem::StiffnessTopOptProblem, topology::AbstractVector)
#     old_grid = problem.ch.dh.grid
#     new_grid = JuAFEM.Grid(old_grid.cells[Bool.(round.(Int, topology))], old_grid.nodes)
#     visualize(new_grid)
# end