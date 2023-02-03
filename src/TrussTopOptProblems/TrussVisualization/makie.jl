using Ferrite
using LinearAlgebra: norm

# using Makie: Makie
using .Makie: Makie
using Makie: lift, cam3d!, Point3f0, Vec3f0, Figure, Auto, linesegments!, Point2f0, RGBAf
using Makie: DataAspect, Axis, LScene, SliderGrid
using ColorSchemes

using ...TopOpt.TopOptProblems: getdim
using ..TrussTopOptProblems: TrussProblem
using ..TopOpt.TopOptProblems.Visualization: _create_colorbar

function visualize(
    problem::TrussProblem{xdim, T};
    u=undef,
    topology=undef,
    undeformed_mesh_color=RGBAf(0,0,0,1.0),
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0,1,1,0.4),
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
    kw...
) where {xdim, T}
    ndim = getdim(problem)
    ncells = Ferrite.getncells(problem)
    nnodes = Ferrite.getnnodes(problem)
    given_u = u!==undef
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
        (label = "element linewidth", range = 0.0:0.01:element_linewidth_range, format = "{:.2f}", startvalue = default_element_linewidth_scale),
        width=Auto(),
    )
    if display_supports
        condition_lsgrid = SliderGrid(
            fig[3, 1],
            (label = "support scale", range = 0.0:0.01:scale_range, format = "{:.2f}", startvalue = default_support_scale),
            (label = "load scale",    range = 0.0:0.01:scale_range, format = "{:.2f}", startvalue = default_load_scale),
            width=Auto(),
        )
    end
    if given_u
        deform_lsgrid = SliderGrid(
            fig[4, 1],
            (label = "deformation exaggeration", range = 0.0:0.01:exagg_range, format = "{:.2f}", startvalue = default_exagg_scale),
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
        _create_colorbar(fig[1,2], colormap, cell_colors)
    end

    # linewidth: 2Xncells vector, 2i ~ 2i-1 represents a line's two endpoints' width
    undeformed_mesh_colors = Vector{RGBAf}(undef, 2*length(topology))
    topology_linewidth = similar(topology, 2*length(topology))
    for i in eachindex(topology)
        ccolor = undeformed_mesh_color
        if cell_colors !== undef
            ccolor = ColorSchemes.get(colormap, scaled_cell_colors[i])
        end
        topology_linewidth[2*i-1:2*i] .= topology[i]
        undeformed_mesh_colors[2*i-1:2*i] .= ccolor
    end
    element_linewidth = lift(s -> topology_linewidth .* s, linewidth_lsgrid.sliders[1].value)
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
                    lift(s -> [PtT(s .* v) for nid in node_ids], condition_lsgrid.sliders[1].value);
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
