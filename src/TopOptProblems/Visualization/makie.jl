import AbstractPlotting
import .Makie
using LinearAlgebra: norm
using AbstractPlotting: lift, cam3d!, Point3f0, Vec3f0, Figure, Auto
using AbstractPlotting.MakieLayout: DataAspect, Axis, labelslidergrid!, set_close_to!,
    labelslider!, LScene
using GeometryBasics: GLTriangleFace
using ..TopOptProblems: getcloaddict

################################
# Credit to Simon Danisch for the conversion code below

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/f16321dee2c77ac9c753fed9b1074a2df7b10db8/src/utilities/utilities.jl#L188
# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L522
function AbstractPlotting.to_vertices(nodes::Vector{<:Ferrite.Node})
    return Point3f0.([n.x for n in nodes])
end

function AbstractPlotting.to_triangles(cells::AbstractVector{<: Ferrite.Cell})
    tris = GLTriangleFace[]
    for cell in cells
        to_triangle(tris, cell)
    end
    tris
end

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L505
function to_triangle(tris, cell::Union{Ferrite.Hexahedron, Ferrite.QuadraticHexahedron})
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

function to_triangle(tris, cell::Union{Ferrite.Tetrahedron, Ferrite.QuadraticTetrahedron})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[3], nodes[2]))
    push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[2]))
    push!(tris, GLTriangleFace(nodes[4], nodes[3], nodes[1]))
    push!(tris, GLTriangleFace(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(tris, cell::Union{Ferrite.Quadrilateral, Ferrite.QuadraticQuadrilateral})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::Union{Ferrite.Triangle, Ferrite.QuadraticTriangle})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
end

function AbstractPlotting.convert_arguments(P, x::AbstractVector{<:Ferrite.Node{N, T}}) where {N, T}
    convert_arguments(P, reinterpret(Point{N, T}, x))
end

################################

function visualize(mesh::Ferrite.Grid{dim, <:Ferrite.AbstractCell, TT}, u; 
        topology=undef, cloaddict=undef,
        undeformed_mesh_color=(:gray, 0.4),
        deformed_mesh_color=(:cyan, 0.4),
        vector_arrowsize=1.0, vector_linewidth=1.0,
        default_support_scale=1.0, default_load_scale=1.0, scale_range=1.0, 
        default_exagg_scale=1.0, exagg_range=1.0) where {dim, TT}
    T = eltype(u)
    nnodes = length(mesh.nodes)
    if topology !== undef
        mesh_cells = mesh.cells[Bool.(round.(Int, topology))]
        # TODO display opacity accroding to topology values
        # undeformed_mesh_color = [(:gray,t) for t in topology]
    else
        mesh_cells = mesh.cells
    end

    # * initialize the makie scene
    fig = Figure(resolution = (1200, 800))

    #TODO make this work without creating a Node
    if dim == 2
        nodes = Vector{Ferrite.Node}(undef, nnodes)
        for (i, node) in enumerate(mesh.nodes)
            nodes[i] = Ferrite.Node((node.x[1], node.x[2], zero(T)))
        end
        u = [u; zeros(T, 1, nnodes)]

        ax1 = Axis(fig[1,1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        nodes = mesh.nodes

        # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
        # https://makie.juliaplots.org/stable/cameras.html#D-Camera
        # ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
        ax1 = LScene(fig[1,1], scenekw = (camera = cam3d!, raw = false), height=750)
    end
    # TODO show the ground mesh in another Axis https://makie.juliaplots.org/stable/makielayout/grids.html
    # ax1.title = "TopOpt result"

    # * support / load appearance / deformatione exaggeration control
    lsgrid = labelslidergrid!(fig,
        ["deformation exaggeration","support scale", "load scale"],
        [LinRange(0.0:0.01:exagg_range), LinRange(0.0:0.01:scale_range), LinRange(0.0:0.01:scale_range)];
        width = Auto(),
        # tellwidth = true,
        # horizontal = false,
    )
    set_close_to!(lsgrid.sliders[1], default_exagg_scale)
    set_close_to!(lsgrid.sliders[2], default_support_scale)
    set_close_to!(lsgrid.sliders[3], default_load_scale)
    fig[2,1] = lsgrid.layout

    # * undeformed mesh
    Makie.mesh!(ax1, nodes, mesh_cells, color = undeformed_mesh_color, shading = true);

    # * deformed mesh
    if norm(u) > eps()
        exagg_deformed_nodes = lift(s -> 
            [Ferrite.Node(Tuple([node.x[j] + s * u[j, i] for j=1:3])) for (i, node) in enumerate(nodes)], 
            lsgrid.sliders[1].value)
        new_nodes = Vector{Ferrite.Node}(undef, length(nodes))
        Makie.mesh!(ax1, exagg_deformed_nodes, mesh_cells, color = deformed_mesh_color)
    end

    # * dot points for deformation nodes
    # Makie.scatter!(ax1, Point3f0.(getfield.(new_nodes, :x)), markersize = 0.1);
    # points = Point3f0.(getfield.(nodes, :x))
    # * deformation vectors
    # GeometryTypes.Vec{3, Float64}
    # displacevec = reinterpret(Vec3f0, u, (size(u, 2),))
    # displacevec = Vec3f0.([u[:,i] for i=1:size(u,2)])
    # displace = norm.(displacevec)
    # Makie.arrows!(points, displacevec, linecolor = (:black, 0.3))

    # TODO pressure loads?
    # * load vectors
    if cloaddict !== undef
        if length(cloaddict) > 0
            loaded_nodes = Point3f0.(nodes[node_ind].x for (node_ind, _) in cloaddict)
            Makie.arrows!(ax1, 
                loaded_nodes,
                lift(s -> Vec3f0.(s .* load_vec for (_, load_vec) in cloaddict), lsgrid.sliders[3].value), 
                linecolor=:purple, arrowcolor=:purple,
                arrowsize=vector_arrowsize, linewidth=vector_linewidth)
            Makie.scatter!(ax1, loaded_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[2].value))
        end
    end

    # * support vectors
    for (nodeset_name, node_ids) in mesh.nodesets
        vectors = []
        if occursin("fixed_u1", nodeset_name)
            push!(vectors, [1.0, 0.0, 0.0])
        elseif occursin("fixed_u2", nodeset_name)
            push!(vectors, [0.0, 1.0, 0.0])
        elseif occursin("fixed_u3", nodeset_name)
            push!(vectors, [0.0, 0.0, 1.0])
        elseif occursin("fixed_all", nodeset_name)
            push!(vectors, [1.0, 0.0, 0.0])
            push!(vectors, [0.0, 1.0, 0.0])
            push!(vectors, [0.0, 0.0, 1.0])
        else
            continue
        end
        fixed_nodes = Point3f0.(nodes[node_ind].x for node_ind in node_ids)
        for v in vectors
            Makie.arrows!(ax1, 
                fixed_nodes,
                lift(s -> [Vec3f0(s .* v) for nid in node_ids], lsgrid.sliders[2].value), 
                linecolor=:orange, arrowcolor=:orange,
                arrowsize=vector_arrowsize, linewidth=vector_linewidth)
        end
        Makie.scatter!(ax1, fixed_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))
    end

    fig
end

"""
draw problem's initial grid with a given displacement vector `u`
"""
function visualize(problem::StiffnessTopOptProblem{dim, T}, u::AbstractVector; 
        kwargs...) where {dim, T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = Ferrite.getnnodes(mesh)
    if u === undef
        node_displacements = zeros(T, dim, nnodes)
    else
        node_displacements = reshape(u[node_dofs], dim, nnodes)
    end
    cloaddict = getcloaddict(problem)
    visualize(mesh, node_displacements; topology=undef, cloaddict=cloaddict, kwargs...)
end

function visualize(problem::StiffnessTopOptProblem{dim, T}; kwargs...) where {dim, T}
    mesh = problem.ch.dh.grid
    nnodes = Ferrite.getnnodes(mesh)
    node_displacements = zeros(T, dim, nnodes)
    cloaddict = getcloaddict(problem)
    visualize(mesh, node_displacements; cloaddict=cloaddict, kwargs...)
end