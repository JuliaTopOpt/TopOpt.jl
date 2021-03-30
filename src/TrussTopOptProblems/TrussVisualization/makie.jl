using AbstractPlotting: lift, cam3d!, Point3f0, Vec3f0, Figure, Auto, linesegments!, Point2f0
using AbstractPlotting.MakieLayout: DataAspect, Axis, labelslidergrid!, set_close_to!,
    labelslider!, LScene
import .Makie

using ...TopOpt.TopOptProblems: getdim
using ..TrussTopOptProblems: TrussProblem, get_fixities_node_set_name
using Ferrite
using LinearAlgebra: norm

"""
    scene, layout = draw_truss_problem(problem; topology=result.topology)
"""
# function visualize(problem::TrussProblem; kwargs...)
#     scene, layout = layoutscene() #resolution = (1200, 900)
#     draw_truss_problem!(scene, layout, problem; kwargs...)
#     display(scene)
#     return scene, layout
# end

function visualize(problem::TrussProblem, u;
    topology=nothing, 
    # stress=nothing, 
    undeformed_mesh_color=(:gray, 1.0),
    deformed_mesh_color=(:cyan, 0.4),
    vector_arrowsize=0.3, vector_linewidth=1.0,
    default_support_scale=1.0, default_load_scale=1.0, scale_range=1.0,
    default_exagg_scale=1.0, exagg_range=10.0,
    default_element_linewidth_scale=6.0, element_linewidth_range=10.0)
    ndim = getdim(problem)
    ncells = Ferrite.getncells(problem)
    nnodes = Ferrite.getnnodes(problem)

    fig = Figure(resolution = (1200, 800))

    if topology !== nothing
        @assert(ncells == length(topology))
        a = reshape([topology topology]', 2*ncells)
        # a ./= maximum(a)
    else
        a = ones(2*ncells)
    end
    # if stress !== nothing
    #     @assert(ncells == length(stress))
    #     q_color = Array{RGBAf0, 1}(undef, length(stress))
    #     for i=1:ncells
    #         if stress[i] < 0
    #             q_color[i] = RGBAf0(0,0,1,0.8)
    #         else
    #             q_color[i] = RGBAf0(1,0,0,0.8)
    #         end
    #     end
    #     color = reshape([q_color q_color]', 2*length(stress))
    # else
    #     color = :black
    # end

    nodes = problem.truss_grid.grid.nodes
    PtT = ndim == 2 ? Point2f0 : Point3f0
    edges_pts = [PtT(nodes[cell.nodes[1]].x) => PtT(nodes[cell.nodes[2]].x) for cell in problem.truss_grid.grid.cells]

    if ndim == 2
        ax1 = Axis(fig[1,1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
        # https://makie.juliaplots.org/stable/cameras.html#D-Camera
        # ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
        ax1 = LScene(fig[1,1], scenekw = (camera = cam3d!, raw = false), height=750)
    end
    # TODO show the ground mesh in another Axis https://makie.juliaplots.org/stable/makielayout/grids.html
    # ax1.title = "Truss TopOpt result"

    # sl1 = layout[2, 1] = LSlider(scene, range = 0.01:0.01:10, startvalue = 1.0)
    lsgrid = labelslidergrid!(fig,
        ["deformation exaggeration", 
         "support scale", "load scale", 
         "element linewidth"],
        [LinRange(0.0:0.01:exagg_range), LinRange(0.0:0.01:scale_range), LinRange(0.0:0.01:scale_range), 
         LinRange(0.0:0.01:element_linewidth_range)];
        width = Auto(),
        tellheight = false,
    )
    set_close_to!(lsgrid.sliders[1], default_exagg_scale)
    set_close_to!(lsgrid.sliders[2], default_support_scale)
    set_close_to!(lsgrid.sliders[3], default_load_scale)
    set_close_to!(lsgrid.sliders[4], default_element_linewidth_scale)
    fig[2,1] = lsgrid.layout

    # * undeformed truss elements
    element_linewidth = lift(s -> a.*s, lsgrid.sliders[4].value)
    linesegments!(ax1, edges_pts, 
                  linewidth = element_linewidth,
                  color = undeformed_mesh_color)

    # * deformed truss elements
    if norm(u) > eps()
        node_dofs = problem.metadata.node_dofs
        @assert length(u) == ndim * nnodes
        exagg_edge_pts = lift(s -> 
            [PtT(nodes[cell.nodes[1]].x) + PtT(u[node_dofs[:,cell.nodes[1]]]*s) => 
             PtT(nodes[cell.nodes[2]].x) + PtT(u[node_dofs[:,cell.nodes[2]]]*s) for cell in problem.truss_grid.grid.cells], 
            lsgrid.sliders[1].value)
        linesegments!(ax1, exagg_edge_pts, 
                      linewidth = element_linewidth,
                      color = deformed_mesh_color)
    end

    # * fixties vectors
    for i=1:ndim
        nodeset_name = get_fixities_node_set_name(i)
        fixed_node_ids = Ferrite.getnodeset(problem.truss_grid.grid, nodeset_name)
        dir = zeros(ndim)
        dir[i] = 1.0
        scaled_base_pts = lift(s->[PtT(nodes[node_id].x) - PtT(dir*s) for node_id in fixed_node_ids], 
            lsgrid.sliders[2].value)
        scaled_fix_dirs = lift(s->fill(PtT(dir*s), length(fixed_node_ids)), lsgrid.sliders[2].value)
        Makie.arrows!(
            ax1,
            scaled_base_pts,
            scaled_fix_dirs,
            arrowcolor=:orange,
            arrowsize=vector_arrowsize,
            linecolor=:orange,
            linewidth=vector_linewidth,
        )
    end
    # * load vectors
    scaled_load_dirs = lift(s->[PtT(force/norm(force)*s) for force in values(problem.force)], 
        lsgrid.sliders[3].value)
    Makie.arrows!(
        ax1,
        [PtT(nodes[node_id].x) for node_id in keys(problem.force)],
        scaled_load_dirs,
        arrowcolor=:purple,
        arrowsize=vector_arrowsize,
        linecolor=:purple,
        linewidth=vector_linewidth,
    )
    fig
end

function visualize(problem::TrussProblem{xdim, T}; kwargs...) where {xdim, T}
    nnodes = Ferrite.getnnodes(problem)
    u = zeros(T, xdim * nnodes)
    visualize(problem, u; kwargs...)
end
