module TopOptMakieExt

using LinearAlgebra: norm
using Makie: Makie, lift, cam3d!, Point3f0, Vec3f0, Figure, Auto, RGBAf
using Makie: DataAspect, Axis, LScene, SliderGrid, linesegments!, Point2f0
using Makie.GeometryBasics
using ColorSchemes
using GeometryBasics: GLTriangleFace
using TopOpt: TopOpt
using TopOpt.TopOptProblems: getcloaddict, boundingbox, getdim, StiffnessTopOptProblem
using TopOpt.TrussTopOptProblems: TrussProblem
using Ferrite

################################
# Credit to Simon Danisch for the conversion code below

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/f16321dee2c77ac9c753fed9b1074a2df7b10db8/src/utilities/utilities.jl#L188
# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L522
function Makie.to_vertices(nodes::Vector{<:Ferrite.Node})
    return Point3f0.([n.x for n in nodes])
end

function Makie.to_triangles(cells::AbstractVector{<:Ferrite.Cell})
    tris = GLTriangleFace[]
    for cell in cells
        to_triangle(tris, cell)
    end
    return tris
end

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L505
function to_triangle(tris, cell::Union{Ferrite.Hexahedron,QuadraticHexahedron})
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
    return push!(tris, GLTriangleFace(nodes[3], nodes[1], nodes[4]))
end

function to_triangle(tris, cell::Union{Ferrite.Tetrahedron,Ferrite.QuadraticTetrahedron})
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[3], nodes[2]))
    push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[2]))
    push!(tris, GLTriangleFace(nodes[4], nodes[3], nodes[1]))
    return push!(tris, GLTriangleFace(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(
    tris, cell::Union{Ferrite.Quadrilateral,Ferrite.QuadraticQuadrilateral}
)
    nodes = cell.nodes
    push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
    return push!(tris, GLTriangleFace(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::Union{Ferrite.Triangle,Ferrite.QuadraticTriangle})
    nodes = cell.nodes
    return push!(tris, GLTriangleFace(nodes[1], nodes[2], nodes[3]))
end

function Makie.convert_arguments(P, x::AbstractVector{<:Ferrite.Node{N,T}}) where {N,T}
    return convert_arguments(P, reinterpret(Point{N,T}, x))
end

"""
Duplicate nodes and cells to make drawing a uniform color per cell face work.
Inspired by: https://discourse.julialang.org/t/makie-triangle-face-colour-mesh/18011/7
"""
function _explode_nodes_and_cells(
    grid::Ferrite.Grid{xdim,cell_type,T}
) where {xdim,cell_type,T}
    new_nodes = Vector{Ferrite.Node}()
    new_cells = similar(grid.cells, 0)
    new_node_id_from_old = Dict{Int,Vector{Int}}(i => [] for i in 1:length(grid.nodes))
    old_node_id_from_new = Vector{Int}()
    node_count = 0
    for (cid, cell) in enumerate(grid.cells)
        for (local_id, nid) in enumerate(cell.nodes)
            if xdim == 3
                push!(new_nodes, grid.nodes[nid])
            elseif xdim == 2
                node = grid.nodes[nid]
                push!(new_nodes, Ferrite.Node((node.x[1], node.x[2], zero(T))))
            else
                error("Unsupported xdim $xdim !")
            end
            push!(new_node_id_from_old[nid], node_count + local_id)
            push!(old_node_id_from_new, nid)
        end
        num_cnodes = length(cell.nodes)
        push!(new_cells, cell_type(Tuple((node_count + 1):(node_count + num_cnodes))))
        node_count += num_cnodes
    end
    @assert length(grid.cells) == length(new_cells)
    return new_nodes, new_cells, new_node_id_from_old, old_node_id_from_new
end

function _create_colorbar(fig, colormap, cell_colors)
    val_range = maximum(cell_colors) - minimum(cell_colors)
    return Makie.Colorbar(
        fig;
        colormap=colormap,
        highclip=:black,
        lowclip=:white,
        ticks=minimum(cell_colors):(val_range / 10):maximum(cell_colors),
        limits=(minimum(cell_colors), maximum(cell_colors)),
    )
end

################################

"""
    function visualize(problem::StiffnessTopOptProblem{dim,T};
        u=undef,
        topology=undef,
        cloaddict=undef,
        undeformed_mesh_color=dim==2 ? RGBAf(0,0,0,1.0) : RGBAf(0.5,0.5,0.5,0.4),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        deformed_mesh_color=RGBAf(0,1,1,0.4),
        vector_arrowsize=1.0,
        vector_linewidth=1.0,
        default_support_scale=1.0,
        default_load_scale=1.0,
        scale_range=1.0,
        default_exagg_scale=1.0,
        exagg_range=10.0,
        kw...
    ) where {dim,T}

Visualizer based on [Makie.jl](https://makie.juliaplots.org/stable/index.html). We take advantage of the interactive
functionality provided by `GLMakie.jl`. To use the interactive backend, please install and activate `GLMakie` by `import Pkg; Pkg.add("GLMakie"); using TopOpt, Makie, GLMakie`

Note that if you want to export publication-quality vector graphics, you can still use `CairoMakie` backend and `save("name.pdf", fig)` with the figure handle return by `visualize`, even though the visualization window does not show up. You can do so by simply replacing `using Makie, GLMakie` with `using Makie, CairoMakie`.
So we recommend using `GLMakie` backend until you are satisfied, and switch backend to export the high-quality figures.

# Inputs

- `problem`: continuum topopt problem

# Optional arguments

- `u=undef`: nodal displacement vector (dim `n_dof`). 
    Usually got from `solver.vars = x_you_want; solver(); u = solver.u;`. If `undef`, assumed to be a zero vector.
- `topology=undef` : desired topology density vector (dim `n_cells`). If `undef`, assume all cells are included. 
    For display, we apply a transparency of `x[i]` to `cell[i]` to see all the gray-scale cells, not only the black and white ones.
- `cloaddict=undef` : Dict(node_id => load vector). If `undef`, the dict will be parsed from the problem by `getcloaddict(problem)`.
- `undeformed_mesh_color` : color used for displaying the undeformed mesh.
- `cell_colors=undef` : Vector (dim `n_cells`) of a value per cell to show the color map. If this is used, `undeformed_mesh_color` will be ignored.
- `draw_legend=false` : draw the color legend for cell_colors.
- `colormap=ColorSchemes.Spectral_10` : color map used to show `cell_color`. See [catalog](https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/) for more options.
- `deformed_mesh_color` : color used for displaying deformed mesh if `u` is specified.
- `vector_arrowsize=1.0` : the vector arrow size used for displaying loads and supports vectors.
- `vector_linewidth=1.0` : line width for loads and supports vectors.
- `default_support_scale=1.0` : the default support scale used in the slider.
- `default_load_scale=1.0` : the default load scale used in the slider.
- `scale_range=1.0` : the upper limit of the sliders controlling the support and load scale sliders.
- `default_exagg_scale=1.0` : default deformation exaggeration scale.
- `exagg_range=10.0` : the upper limit of the slider controlling the deformation exaggeration slider.
- `kw...` : optional keyword argument passed to [Makie.mesh!](https://docs.makie.org/stable/api/#mesh!) function.

# Returns
- `Makie.Figure` handle

"""
function TopOpt.visualize(
    problem::StiffnessTopOptProblem{dim,T};
    u=undef,
    topology=undef,
    cloaddict=undef,
    undeformed_mesh_color=dim == 2 ? RGBAf(0, 0, 0, 1.0) : RGBAf(0.5, 0.5, 0.5, 0.4),
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
    display_supports=true,
    vector_arrowsize=1.0,
    vector_linewidth=1.0,
    default_support_scale=1.0,
    default_load_scale=1.0,
    scale_range=1.0,
    default_exagg_scale=1.0,
    exagg_range=10.0,
    kw...,
) where {dim,T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = Ferrite.getnnodes(mesh)
    # coord_min, coord_max = boundingbox(mesh)
    # mesh_dim = maximum([coord_max...] - [coord_min...])

    given_u = u !== undef
    cloaddict = cloaddict === undef ? getcloaddict(problem) : cloaddict

    mesh_cells = mesh.cells
    topology = topology == undef ? ones(T, length(mesh_cells)) : topology
    nodes = Vector{Ferrite.Node}(undef, nnodes)
    if dim == 2
        for (i, node) in enumerate(mesh.nodes)
            nodes[i] = Ferrite.Node((node.x[1], node.x[2], zero(T)))
        end
    else
        nodes = mesh.nodes
    end

    # * initialize the makie scene
    fig = Figure()

    if dim == 2
        ax1 = Axis(fig[1, 1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
        # https://makie.juliaplots.org/stable/cameras.html#D-Camera
        # ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
        ax1 = LScene(fig[1, 1]; scenekw=(camera=cam3d!, raw=false)) # , height=750
    end

    # * support / load appearance / deformatione exaggeration control
    if display_supports
        condition_lsgrid = SliderGrid(
            fig[2, 1],
            (
                label="support scale",
                range=0.0:0.01:scale_range,
                format="{:.2f}",
                startvalue=default_support_scale,
            ),
            (
                label="load scale",
                range=0.0:0.01:scale_range,
                format="{:.2f}",
                startvalue=default_load_scale,
            );
            width=Auto(),
        )
    end
    if given_u
        deform_lsgrid = SliderGrid(
            fig[3, 1],
            (
                label="deformation exaggeration",
                range=0.0:0.01:exagg_range,
                format="{:.2f}",
                startvalue=default_exagg_scale,
            );
            width=Auto(),
        )
    end

    dup_nodes, dup_cells, new_node_id_from_old, old_node_id_from_new = _explode_nodes_and_cells(
        mesh
    )
    # each color for each duplicated vertex
    undeformed_mesh_colors = Vector{RGBAf}(undef, length(dup_nodes))
    # * color per cell
    scaled_cell_colors = similar(topology)
    scaled_cell_colors .= 0.0
    if cell_colors !== undef
        @assert length(cell_colors) == length(topology)
        val_range = maximum(cell_colors) - minimum(cell_colors)
        scaled_cell_colors = (cell_colors .- minimum(cell_colors)) / val_range
    end
    for i in eachindex(dup_cells)
        cell_xvar = topology[i]
        for new_nid in dup_cells[i].nodes
            # set the alpha value to pseudo density of the cell
            ccolor = undeformed_mesh_color
            if cell_colors !== undef
                ccolor = ColorSchemes.get(colormap, scaled_cell_colors[i])
            end
            undeformed_mesh_colors[new_nid] = RGBAf(ccolor.r, ccolor.g, ccolor.b, cell_xvar)
        end
    end
    if cell_colors !== undef && draw_legend
        _create_colorbar(fig[1, 2], colormap, cell_colors)
    end

    # * Undeformed mesh
    Makie.mesh!(ax1, dup_nodes, dup_cells; color=undeformed_mesh_colors, kw...)

    # * deformed mesh
    if given_u
        if u !== undef
            u = reshape(u[node_dofs], dim, nnodes)
            if dim == 2
                u = [u; zeros(T, 1, nnodes)]
            end
        end
        dup_u = Matrix{T}(undef, 3, length(dup_nodes))
        for new_nid in axes(dup_u, 2)
            dup_u[:, new_nid] = u[:, old_node_id_from_new[new_nid]]
        end

        exagg_deformed_nodes = lift(
            s -> [
                Ferrite.Node(
                    Tuple([new_node.x[ax_id] + s * dup_u[ax_id, nid] for ax_id in 1:3])
                ) for (nid, new_node) in enumerate(dup_nodes)
            ],
            deform_lsgrid.sliders[1].value,
        )
        deformed_mesh_colors = [
            RGBAf(
                deformed_mesh_color.r,
                deformed_mesh_color.g,
                deformed_mesh_color.b,
                ccolor.alpha,
            ) for ccolor in undeformed_mesh_colors
        ]
        Makie.mesh!(ax1, exagg_deformed_nodes, dup_cells; color=deformed_mesh_colors, kw...)
    end

    if display_supports
        # TODO pressure loads?
        # * load vectors
        if cloaddict !== undef
            if length(cloaddict) > 0
                loaded_nodes = Point3f0.(nodes[node_ind].x for (node_ind, _) in cloaddict)
                Makie.arrows!(
                    ax1,
                    loaded_nodes,
                    lift(
                        s -> Vec3f0.(s .* load_vec for (_, load_vec) in cloaddict),
                        condition_lsgrid.sliders[2].value,
                    );
                    linecolor=:purple,
                    arrowcolor=:purple,
                    arrowsize=vector_arrowsize,
                    linewidth=vector_linewidth,
                )
                Makie.scatter!(ax1, loaded_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[2].value))
            end
        end

        # * support vectors
        ch = problem.ch
        for (dbc_id, dbc) in enumerate(ch.dbcs)
            support_vectors = []
            if 1 in dbc.components
                push!(support_vectors, [1.0, 0.0, 0.0])
            end
            if 2 in dbc.components
                push!(support_vectors, [0.0, 1.0, 0.0])
            end
            if 3 in dbc.components
                push!(support_vectors, [0.0, 0.0, 1.0])
            end
            node_ids = dbc.faces
            fixed_nodes = Point3f0.(nodes[node_ind].x for node_ind in node_ids)
            # draw one axis for all nodes in the set each time
            for v in support_vectors
                Makie.arrows!(
                    ax1,
                    fixed_nodes,
                    lift(
                        s -> [Vec3f0(s .* v) for nid in node_ids],
                        condition_lsgrid.sliders[1].value,
                    );
                    linecolor=:orange,
                    arrowcolor=:orange,
                    arrowsize=vector_arrowsize,
                    linewidth=vector_linewidth,
                )
            end
            Makie.scatter!(ax1, fixed_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))
        end
    end # end if display_supports

    return fig
end

function TopOpt.visualize(
    problem::TrussProblem{xdim,T};
    u=undef,
    topology=undef,
    undeformed_mesh_color=RGBAf(0, 0, 0, 1.0),
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
    display_supports=true,
    vector_arrowsize=0.3,
    vector_linewidth=1.0,
    default_support_scale=1e-2,
    default_load_scale=1e-2,
    scale_range=1.0,
    default_exagg_scale=1.0,
    exagg_range=10.0,
    default_element_linewidth_scale=6.0,
    element_linewidth_range=10.0,
    kw...,
) where {xdim,T}
    ndim = getdim(problem)
    ncells = Ferrite.getncells(problem)
    nnodes = Ferrite.getnnodes(problem)
    given_u = u !== undef
    topology = topology == undef ? ones(T, ncells) : topology

    fig = Figure()
    if ndim == 2
        ax1 = Axis(fig[1, 1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        ax1 = LScene(fig[1, 1]; scenekw=(camera=cam3d!, raw=false)) #, height=750)
    end

    # * linewidth scaling / support / load appearance / deformatione exaggeration control
    linewidth_lsgrid = SliderGrid(
        fig[2, 1],
        (
            label="element linewidth",
            range=0.0:0.01:element_linewidth_range,
            format="{:.2f}",
            startvalue=default_element_linewidth_scale,
        );
        width=Auto(),
    )
    if display_supports
        condition_lsgrid = SliderGrid(
            fig[3, 1],
            (
                label="support scale",
                range=0.0:0.01:scale_range,
                format="{:.2f}",
                startvalue=default_support_scale,
            ),
            (
                label="load scale",
                range=0.0:0.01:scale_range,
                format="{:.2f}",
                startvalue=default_load_scale,
            );
            width=Auto(),
        )
    end
    if given_u
        deform_lsgrid = SliderGrid(
            fig[4, 1],
            (
                label="deformation exaggeration",
                range=0.0:0.01:exagg_range,
                format="{:.2f}",
                startvalue=default_exagg_scale,
            );
            width=Auto(),
        )
    end

    # * undeformed truss elements
    nodes = problem.truss_grid.grid.nodes
    PtT = ndim == 2 ? Point2f0 : Point3f0
    edges_pts = [
        PtT(nodes[cell.nodes[1]].x) => PtT(nodes[cell.nodes[2]].x) for
        cell in problem.truss_grid.grid.cells
    ]

    # * linewidth and color per cell
    scaled_cell_colors = similar(topology)
    scaled_cell_colors .= 0.0
    if cell_colors !== undef
        @assert length(cell_colors) == length(topology) "$(length(cell_colors)) , $(length(topology))"
        val_range = maximum(cell_colors) - minimum(cell_colors)
        scaled_cell_colors = (cell_colors .- minimum(cell_colors)) / val_range
    end
    if cell_colors !== undef && draw_legend
        _create_colorbar(fig[1, 2], colormap, cell_colors)
    end

    # linewidth: 2Xncells vector, 2i ~ 2i-1 represents a line's two endpoints' width
    undeformed_mesh_colors = Vector{RGBAf}(undef, 2 * length(topology))
    topology_linewidth = similar(topology, 2 * length(topology))
    for i in eachindex(topology)
        ccolor = undeformed_mesh_color
        if cell_colors !== undef
            ccolor = ColorSchemes.get(colormap, scaled_cell_colors[i])
        end
        topology_linewidth[(2 * i - 1):(2 * i)] .= topology[i]
        undeformed_mesh_colors[(2 * i - 1):(2 * i)] .= ccolor
    end
    element_linewidth = lift(
        s -> topology_linewidth .* s, linewidth_lsgrid.sliders[1].value
    )
    linesegments!(ax1, edges_pts; linewidth=element_linewidth, color=undeformed_mesh_colors)

    # # * deformed truss elements
    if given_u
        node_dofs = problem.metadata.node_dofs
        @assert length(u) == ndim * nnodes
        exagg_edge_pts = lift(
            s -> [
                PtT(nodes[cell.nodes[1]].x) + PtT(u[node_dofs[:, cell.nodes[1]]] * s) =>
                    PtT(nodes[cell.nodes[2]].x) + PtT(u[node_dofs[:, cell.nodes[2]]] * s) for
                cell in problem.truss_grid.grid.cells
            ],
            deform_lsgrid.sliders[1].value,
        )
        linesegments!(
            ax1, exagg_edge_pts; linewidth=element_linewidth, color=deformed_mesh_color
        )
    end

    if display_supports
        # * load vectors
        loaded_nodes = [PtT(nodes[node_id].x) for node_id in keys(problem.force)]
        scaled_load_dirs = lift(
            s -> [PtT(force / norm(force) * s) for force in values(problem.force)],
            condition_lsgrid.sliders[2].value,
        )
        Makie.arrows!(
            ax1,
            loaded_nodes,
            scaled_load_dirs;
            arrowcolor=:purple,
            arrowsize=vector_arrowsize,
            linecolor=:purple,
            linewidth=vector_linewidth,
        )
        Makie.scatter!(ax1, loaded_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))

        # * fixties vectors
        ch = problem.ch
        for (dbc_id, dbc) in enumerate(ch.dbcs)
            support_vectors = []
            node_ids = dbc.faces
            if 1 in dbc.components
                push!(support_vectors, [1.0, 0.0, 0.0])
            end
            if 2 in dbc.components
                push!(support_vectors, [0.0, 1.0, 0.0])
            end
            if 3 in dbc.components
                push!(support_vectors, [0.0, 0.0, 1.0])
            end
            fixed_nodes = PtT.(nodes[node_ind].x for node_ind in node_ids)
            for v in support_vectors
                Makie.arrows!(
                    fixed_nodes,
                    lift(
                        s -> [PtT(s .* v) for nid in node_ids],
                        condition_lsgrid.sliders[1].value,
                    );
                    arrowcolor=:orange,
                    arrowsize=vector_arrowsize,
                    linecolor=:orange,
                    linewidth=vector_linewidth,
                )
            end
            Makie.scatter!(ax1, fixed_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))
        end
    end # end if display_supports

    return fig
end

end
