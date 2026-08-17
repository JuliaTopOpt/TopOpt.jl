module TopOptMakieExt

using LinearAlgebra: norm
using Random: rand
using Makie: Makie, lift, cam3d!, Point3f, Vec3f, Figure, Auto, RGBAf
using Makie: DataAspect, Axis, LScene, SliderGrid, linesegments!, Point2f
using Makie: ColorSchemes
using Makie: current_backend, update_cam!, Mouse, Orthographic, Perspective, on, Observable
using Makie.Observables: throttle
using GeometryBasics
using GeometryBasics: TriangleFace
using TopOpt: TopOpt
using TopOpt.TopOptProblems:
    getcloaddict,
    boundingbox,
    getdim,
    AbstractTopOptProblem,
    StiffnessTopOptProblem,
    HeatTransferTopOptProblem,
    HeatConductionProblem,
    PointLoadCantilever
using TopOpt.TrussTopOptProblems: TrussProblem
using Ferrite

################################
# Credit to Simon Danisch for the conversion code below

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/f16321dee2c77ac9c753fed9b1074a2df7b10db8/src/utilities/utilities.jl#L188
# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L522
function Makie.to_vertices(nodes::Vector{<:Ferrite.Node})
    return Point3f.([n.x for n in nodes])
end

function Makie.to_triangles(cells::AbstractVector{<:Ferrite.AbstractCell})
    # Ferrite 1.x: concrete cell types (Quadrilateral, Hexahedron, ...) are
    # subtypes of AbstractCell, not the legacy `Cell` alias, so dispatch on
    # AbstractCell to cover all cell kinds.
    tris = TriangleFace{Int}[]
    for cell in cells
        to_triangle(tris, cell)
    end
    return tris
end

# https://github.com/JuliaPlots/AbstractPlotting.jl/blob/444813136a506eba8b5b03e2125c7a5f24e825cb/src/conversions.jl#L505
function to_triangle(tris, cell::Union{Ferrite.Hexahedron,QuadraticHexahedron})
    nodes = cell.nodes
    push!(tris, TriangleFace{Int}(nodes[1], nodes[2], nodes[5]))
    push!(tris, TriangleFace{Int}(nodes[5], nodes[2], nodes[6]))

    push!(tris, TriangleFace{Int}(nodes[6], nodes[2], nodes[3]))
    push!(tris, TriangleFace{Int}(nodes[3], nodes[6], nodes[7]))

    push!(tris, TriangleFace{Int}(nodes[7], nodes[8], nodes[3]))
    push!(tris, TriangleFace{Int}(nodes[3], nodes[8], nodes[4]))

    push!(tris, TriangleFace{Int}(nodes[4], nodes[8], nodes[5]))
    push!(tris, TriangleFace{Int}(nodes[5], nodes[4], nodes[1]))

    push!(tris, TriangleFace{Int}(nodes[1], nodes[2], nodes[3]))
    return push!(tris, TriangleFace{Int}(nodes[3], nodes[1], nodes[4]))
end

function to_triangle(tris, cell::Union{Ferrite.Tetrahedron,Ferrite.QuadraticTetrahedron})
    nodes = cell.nodes
    push!(tris, TriangleFace{Int}(nodes[1], nodes[3], nodes[2]))
    push!(tris, TriangleFace{Int}(nodes[3], nodes[4], nodes[2]))
    push!(tris, TriangleFace{Int}(nodes[4], nodes[3], nodes[1]))
    return push!(tris, TriangleFace{Int}(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(
    tris, cell::Union{Ferrite.Quadrilateral,Ferrite.QuadraticQuadrilateral}
)
    nodes = cell.nodes
    push!(tris, TriangleFace{Int}(nodes[1], nodes[2], nodes[3]))
    return push!(tris, TriangleFace{Int}(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::Union{Ferrite.Triangle,Ferrite.QuadraticTriangle})
    nodes = cell.nodes
    return push!(tris, TriangleFace{Int}(nodes[1], nodes[2], nodes[3]))
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
    for (_, cell) in enumerate(grid.cells)
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

"""
Surface texture helpers: CPU per-vertex stripe / hatch / crosshatch overlay
on top of the existing density-colored vertices. Cross-backend safe (no UVs,
no GLSL shaders): every vertex gets a `0` (in mark) or `1` (in gap) decision
and the base color is linearly blended toward `texture_color` by `strength`.

Pattern is oriented in the dominant 2D plane (perpendicular to the longest
world axis of the mesh), so stripes and hatches align with the natural framing
of the part instead of being tied to whatever angle the user happened to pick.
"""
function _dominant_plane_axes(span::NTuple{3,T}) where {T<:Real}
    sx, sy, sz = span
    if sx >= sy && sx >= sz
        return (Vec3f(0, 1, 0), Vec3f(0, 0, 1))
    elseif sy >= sx && sy >= sz
        return (Vec3f(1, 0, 0), Vec3f(0, 0, 1))
    else
        return (Vec3f(1, 0, 0), Vec3f(0, 1, 0))
    end
end

# Square-wave on a 1D phase: 0 on the first half-period, 1 on the second.
function _rect_wave(x::Real, period::Real)
    return mod(x, period) < (period / 2) ? 0.0 : 1.0
end

# Compute the texture modulation at a world-space point. Returns 0 (in mark)
# or 1 (in gap); the caller decides how to blend. `plane_axes` must be two
# orthogonal unit vectors spanning the dominant plane.
function _texture_modulation(
    p::Vec3f, pattern::Symbol, plane_axes::NTuple{2,Vec3f}, period::Real, angle_deg::Real
)
    a1, a2 = plane_axes
    pa = dot(p, a1)
    pb = dot(p, a2)
    c = cosd(angle_deg)
    s = sind(angle_deg)
    if pattern === :stripes
        return _rect_wave(pa, period)
    elseif pattern === :hatch
        return _rect_wave(pa * c + pb * s, period)
    elseif pattern === :crosshatch
        return max(
            _rect_wave(pa * c + pb * s, period), _rect_wave(-pa * s + pb * c, period)
        )
    else
        throw(
            ArgumentError(
                "unsupported surface_texture pattern $(repr(pattern)); " *
                "valid: :none, :stripes, :hatch, :crosshatch",
            ),
        )
    end
end

"""
Resolve `texture_period` from the mesh bounding box: one tenth of the
longest world span, which yields ~10 stripes/cycles across a nominally-sized
optimized part and reads cleanly at the default figure size. Pass a finite
value to override.
"""
function _auto_texture_period(bbox_lo::NTuple{dim,T}, bbox_hi::NTuple{dim,T}) where {dim,T}
    longest = if dim == 2
        max(bbox_hi[1] - bbox_lo[1], bbox_hi[2] - bbox_lo[2])
    else
        max(bbox_hi[1] - bbox_lo[1], bbox_hi[2] - bbox_lo[2], bbox_hi[3] - bbox_lo[3])
    end
    # Finer period than the rough 1/10 default gives ~20 stripes across
    # the mesh — enough detail to read as a contour overlay rather than a
    # chunky band pattern.
    return max(longest / 20, eps(T))
end

# In-place per-vertex color modulation. `:none` short-circuits to a no-op so
# callers don't need to gate on the symbol themselves.
function _apply_surface_texture!(
    colors::Vector{RGBAf},
    nodes::Vector{Ferrite.Node},
    pattern::Symbol,
    period::Real,
    angle_deg::Real,
    texture_color::RGBAf,
    strength::Real,
    bbox_lo::NTuple{3,T},
    bbox_hi::NTuple{3,T},
) where {T<:Real}
    pattern === :none && return nothing
    span = (bbox_hi[1] - bbox_lo[1], bbox_hi[2] - bbox_lo[2], bbox_hi[3] - bbox_lo[3])
    plane_axes = _dominant_plane_axes(span)
    center = (
        T(0.5) * (bbox_lo[1] + bbox_hi[1]),
        T(0.5) * (bbox_lo[2] + bbox_hi[2]),
        T(0.5) * (bbox_lo[3] + bbox_hi[3]),
    )
    sr = clamp(Float64(strength), 0.0, 1.0)
    for i in eachindex(colors, nodes)
        # Cells below the density threshold are fully transparent (alpha=0);
        # skipping them keeps the marker from drawing solid black on void
        # space, which otherwise makes the texture look like surface dust
        # rather than a contour band.
        colors[i].alpha < 1e-3 && continue
        nx = T(nodes[i].x[1])
        ny = T(dim_xyz(nodes[i])[2])
        nz = T(dim_xyz(nodes[i])[3])
        mod_on = _texture_modulation(
            Vec3f(nx - center[1], ny - center[2], nz - center[3]),
            pattern,
            plane_axes,
            period,
            angle_deg,
        )
        mod_on < 0.5 || continue
        c = colors[i]
        # Modulating toward texture_color while preserving alpha keeps the
        # resulting band readable on both opaque solid voxels and partially
        # translucent boundary voxels.
        colors[i] = RGBAf(
            (1 - sr) * c.r + sr * texture_color.r,
            (1 - sr) * c.g + sr * texture_color.g,
            (1 - sr) * c.b + sr * texture_color.b,
            c.alpha,
        )
    end
    return nothing
end

# Cross-dim accessor for ferrite nodes (always 3D in the exploded mesh; 2D
# problems supply a synthetic z=0 from `_explode_nodes_and_cells`).
dim_xyz(n::Ferrite.Node) = (n.x[1], n.x[2], n.x[3])

"""
Emit a CSS color string from an `RGBAf`. `RGBAf(0.6, 0.6, 0.6, 0.85)` is
NOT valid CSS — `string(::RGBAf)` would emit `RGBAf{Float32}(0.6, …)`
and the swatch paints blank. This helper emits the 8-bit `rgb(r, g, b)` /
`rgba(r, g, b, a)` form browsers actually accept.
"""
function css_color(c::Makie.RGBAf)
    r8 = round(Int, clamp(c.r, 0.0, 1.0) * 255)
    g8 = round(Int, clamp(c.g, 0.0, 1.0) * 255)
    b8 = round(Int, clamp(c.b, 0.0, 1.0) * 255)
    a = clamp(Float64(c.alpha), 0.0, 1.0)
    return a >= 1.0 ? "rgb($r8, $g8, $b8)" : "rgba($r8, $g8, $b8, $(round(a; digits=3)))"
end

"""
Wire the mesh's lighting model. `:none` (flat) gives the existing
NoShading coloring, which keeps the density alpha and texture readable
but shows no surface curvature. `:default` switches to FastShading with
ambient + directional lighting + a Phong-style material, so the mesh
reads as a real surface — the typical engineering "rendered metal" look.
"""
function _setup_lighting!(ax, lighting::Symbol)
    if lighting === :none
        return nothing
    elseif lighting === :default
        scene = ax.scene
        if !any(l -> l isa Makie.AmbientLight, scene.lights)
            push!(scene.lights, Makie.AmbientLight(RGBf(0.95, 0.96, 1.0)))
        end
        if !any(l -> l isa Makie.DirectionalLight, scene.lights)
            push!(
                scene.lights,
                Makie.DirectionalLight(RGBf(0.95, 0.95, 0.9), Vec3f(-0.2, -0.3, -1.0)),
            )
        end
        return nothing
    else
        throw(
            ArgumentError(
                "unsupported lighting mode $(repr(lighting)); valid: :none, :default"
            ),
        )
    end
end

"""
Resolve the shading kwarg to pass to `mesh!` based on the chosen
lighting mode. `:default` returns `FastShading` (cross-backend, picks up
the ambient + directional lights added by `_setup_lighting!`); `:none`
returns `NoShading` to keep the per-vertex coloring unchanged by lighting.
"""
function _shading_for(lighting::Symbol)
    return lighting === :default ? Makie.FastShading : Makie.NoShading
end

"""
Returns true when the active Makie backend is CairoMakie. CairoMakie
produces static vector/raster output and does not implement the sliders
or interactive camera controls used by `_setup_*` helpers. Callers
auto-disable `interactive` when this returns true.
"""
_is_cairo_backend() = occursin("Cairo", string(current_backend()))

"""
Compute the bounding-box diagonal (characteristic length) of a set of
nodes. Used to auto-scale arrow sizes, linewidths, and slider defaults
so visualizations look reasonable regardless of problem size.
"""
function _mesh_scale(nodes, dim)
    if dim == 2
        xs = [n.x[1] for n in nodes]
        ys = [n.x[2] for n in nodes]
        return sqrt((maximum(xs) - minimum(xs))^2 + (maximum(ys) - minimum(ys))^2)
    else
        xs = [n.x[1] for n in nodes]
        ys = [n.x[2] for n in nodes]
        zs = [n.x[3] for n in nodes]
        return sqrt(
            (maximum(xs) - minimum(xs))^2 +
            (maximum(ys) - minimum(ys))^2 +
            (maximum(zs) - minimum(zs))^2,
        )
    end
end

"""
Return the same dim-aware default undeformed-mesh color used by the
Stiffness and HeatTransfer `visualize` calls: lighter gray for 2D problems,
slightly translucent lighter gray for 3D. Used by the static viewer to
keep the legend swatches consistent with the rendered mesh without
requiring the caller to thread the color through `kw...`.
"""
function _default_und_mesh_color(problem)
    dim = getdim(problem)
    return dim == 2 ? RGBAf(0.35, 0.35, 0.35, 1.0) : RGBAf(0.6, 0.6, 0.6, 0.85)
end

"""
Derive sensible default arrow parameters from the mesh characteristic
length L (the bounding-box diagonal).
"""
function _auto_arrow_params(L, dim)
    arrow_size = clamp(0.08 * L, 0.01, 100.0)
    arrow_linewidth = dim == 2 ? 2.0 : max(arrow_size / 56, 0.016)
    default_scale = clamp(1.0, 0.001, 1000.0)
    scale_range = clamp(20.0 * default_scale, 0.1, 10000.0)
    arrow_quality = 20
    return (; arrow_size, arrow_linewidth, default_scale, scale_range, arrow_quality)
end

function _plot_voxel_edges!(ax, nodes, cells, colors)
    edges = (
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 5),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
    )
    segments = Point3f[]
    for cell in cells
        ids = cell.nodes
        length(ids) == 8 || continue
        maximum(colors[id].alpha for id in ids) < 1e-3 && continue
        for (i, j) in edges
            push!(segments, Point3f(nodes[ids[i]].x...))
            push!(segments, Point3f(nodes[ids[j]].x...))
        end
    end
    isempty(segments) || Makie.linesegments!(
        ax,
        segments;
        color=RGBAf(0.08, 0.14, 0.24, 0.5),
        linewidth=0.8,
        transparency=true,
    )
    return nothing
end

"""
Plot arrows with consistent aesthetics across 2D and 3D. Picks
`arrows2d!` / `arrows3d!` based on `dim`, then overlays a small scatter
marker at each arrow's tail so the origin is visible on a flat 2D
background.

Makie's arrows API differs by dim: 2D arrows take `shaftwidth`
(pixel-space) and use thin defaults; 3D arrows take `shaftradius`
(world-space) and default to 0.05. The `arrow_linewidth` argument is
mapped to whichever knob is valid.
"""
function _plot_arrows!(
    ax, points, directions; arrow_color, arrow_linewidth, arrow_quality, arrow_size, dim
)
    if dim == 2
        Makie.arrows2d!(
            ax,
            [Point2f(p[1], p[2]) for p in points],
            directions;
            color=arrow_color,
            shaftwidth=arrow_linewidth,
            lengthscale=arrow_size,
            overdraw=true,
            depth_shift=-1.0f-3,
        )
    else
        Makie.arrows3d!(
            ax,
            points,
            directions;
            color=arrow_color,
            quality=arrow_quality,
            lengthscale=arrow_size,
            shaftradius=arrow_linewidth,
            overdraw=true,
            depth_shift=-1.0f-3,
        )
    end
    Makie.scatter!(
        ax,
        points;
        color=arrow_color,
        markersize=dim == 2 ? 6 : 7,
        strokecolor=:white,
        strokewidth=1.0,
        overdraw=true,
        depth_shift=-1.0f-3,
    )
    return nothing
end

"""
Add a legend identifying the undeformed mesh, load arrows, and support
arrows. Skipped on the CairoMakie backend (Legend
blocks render inconsistently in PDF/SVG exports).
"""
function _add_legend!(
    fig,
    undeformed_mesh_color,
    load_arrow_color,
    support_arrow_color;
    has_colorbar::Bool=false,
    plot_height=387,
)
    _is_cairo_backend() && return fig
    elements = [
        Makie.PolyElement(; color=undeformed_mesh_color),
        Makie.MarkerElement(; marker=:circle, color=load_arrow_color),
        Makie.MarkerElement(; marker=:circle, color=support_arrow_color),
    ]
    labels = ["undeformed mesh", "load arrows", "support arrows"]
    legend_slot = has_colorbar ? fig[1, 3] : fig[1, 2]
    Makie.Legend(
        legend_slot,
        elements,
        labels;
        tellheight=false,
        tellwidth=true,
        framevisible=true,
        backgroundcolor=RGBAf(1, 1, 1, 0.85),
        halign=:left,
        valign=:top,
        margin=(0, 0, 0, -8),
        labelsize=11,
        patchsize=(12, 12),
        rowgap=3,
        padding=(4, 4, 4, 4),
    )
    if has_colorbar
        Makie.colsize!(fig.layout, 1, Makie.Auto())
        Makie.colsize!(fig.layout, 2, Makie.Fixed(60))
        Makie.colsize!(fig.layout, 3, Makie.Fixed(150))
    else
        Makie.colsize!(fig.layout, 1, Makie.Auto())
        Makie.colsize!(fig.layout, 2, Makie.Fixed(150))
    end
    Makie.rowsize!(fig.layout, 1, Makie.Fixed(plot_height))
    return fig
end

################################

"""
    function visualize(problem::StiffnessTopOptProblem{dim,T};
        static=false,
        u=undef,
        topology=undef,
        cloaddict=undef,
        undeformed_mesh_color=dim == 2 ? RGBAf(0.35, 0.35, 0.35, 1.0) :
            RGBAf(0.6, 0.6, 0.6, 0.85),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
        surface_texture=:crosshatch,
        texture_period=Auto,
        texture_angle=45.0,
        texture_color=RGBAf(0.18, 0.26, 0.4, 1.0),
        texture_strength=0.06,
        density_threshold=0.5,
        alpha_from_density=true,
        interactive=true,
        display_supports=true,
        lighting=:none,
        vector_arrowsize=Auto,
        load_arrow_color=RGBAf(0.72, 0.12, 0.1, 1.0),
        support_arrow_color=RGBAf(0.72, 0.5, 0.02, 1.0),
        load_arrow_linewidth=Auto,
        support_arrow_linewidth=Auto,
        arrow_quality=Auto,
        default_support_scale=Auto,
        default_load_scale=Auto,
        scale_range=Auto,
        default_exagg_scale=Auto,
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

- `static=false` : when `true`, return a `Bonito.App` with client-side camera controls instead of a `Makie.Figure`.
- `u=undef`: nodal displacement vector (dim `n_dof`). 
    Usually got from `solver.vars = x_you_want; solver(); u = solver.u;`. If `undef`, assumed to be a zero vector.
- `topology=undef` : desired topology density vector (dim `n_cells`). If `undef`, assume all cells are included. 
    For display, we apply a transparency of `x[i]` to `cell[i]` to see all the gray-scale cells, not only the black and white ones.
- `cloaddict=undef` : Dict(node_id => load vector). If `undef`, the dict will be parsed from the problem by `getcloaddict(problem)`.
- `support_spec=nothing` : override the support arrows with a vector of `(component, node_ids)` pairs; `nothing` draws the problem's own Dirichlet BCs.
- `undeformed_mesh_color` : color used for displaying the undeformed mesh.
- `cell_colors=undef` : Vector (dim `n_cells`) of a value per cell to show the color map. If this is used, `undeformed_mesh_color` will be ignored.
- `draw_legend=false` : draw the color legend for cell_colors.
- `colormap=ColorSchemes.Spectral_10` : color map used to show `cell_color`. See [catalog](https://juliagraphics.github.io/ColorSchemes.jl/stable/catalogue/) for more options.
- `deformed_mesh_color` : color used for displaying deformed mesh if `u` is specified.
- `display_supports=true` : draw the support (Dirichlet BC) markers and arrows.
- `lighting=:none` : lighting mode for the 3D scene (`:none` or `:default`).
- `vector_arrowsize=Auto` : the vector arrow size used for displaying loads and supports vectors.
- `default_support_scale=Auto` : the default support scale used in the slider.
- `default_load_scale=Auto` : the default load scale used in the slider.
- `scale_range=Auto` : the upper limit of the sliders controlling the support and load scale sliders.
- `default_exagg_scale=Auto` : default deformation exaggeration scale.
- `exagg_range=10.0` : the upper limit of the slider controlling the deformation exaggeration slider.
- `kw...` : optional keyword argument passed to [Makie.mesh!](https://docs.makie.org/stable/api/#mesh!) function.

# Returns
- `Makie.Figure` handle

"""
function TopOpt.visualize(
    problem::StiffnessTopOptProblem{dim,T};
    static=false,
    u=undef,
    topology=undef,
    cloaddict=undef,
    support_spec=nothing,
    undeformed_mesh_color=if dim == 2
        RGBAf(0.35, 0.35, 0.35, 1.0)
    else
        RGBAf(0.6, 0.6, 0.6, 0.85)
    end,
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
    surface_texture=:crosshatch,
    texture_period=Auto,
    texture_angle=45.0,
    texture_color=RGBAf(0.18, 0.26, 0.4, 1.0),
    texture_strength=0.06,
    density_threshold=0.5,
    alpha_from_density=true,
    interactive=true,
    display_supports=true,
    lighting=:none,
    vector_arrowsize=Auto,
    load_arrow_color=RGBAf(0.72, 0.12, 0.1, 1.0),
    support_arrow_color=RGBAf(0.72, 0.5, 0.02, 1.0),
    load_arrow_linewidth=Auto,
    support_arrow_linewidth=Auto,
    arrow_quality=Auto,
    default_support_scale=Auto,
    default_load_scale=Auto,
    scale_range=Auto,
    default_exagg_scale=Auto,
    exagg_range=10.0,
    kw...,
) where {dim,T}
    if static
        return TopOpt._static_visualization(
            problem;
            u,
            topology,
            cloaddict,
            support_spec,
            undeformed_mesh_color,
            cell_colors,
            draw_legend=true,
            colormap,
            deformed_mesh_color,
            surface_texture,
            texture_period,
            texture_angle,
            texture_color,
            texture_strength,
            density_threshold,
            alpha_from_density,
            interactive=false,
            display_supports,
            lighting,
            vector_arrowsize,
            load_arrow_color,
            support_arrow_color,
            load_arrow_linewidth,
            support_arrow_linewidth,
            arrow_quality,
            default_support_scale,
            default_load_scale,
            scale_range,
            default_exagg_scale,
            exagg_range,
            kw...,
        )
    end
    # CairoMakie produces static output: sliders, interactive camera
    # controls, and the legend have no effect there. Auto-disable so users
    # who activate CairoMakie don't have to remember to pass `interactive=false`.
    interactive::Bool = interactive && !_is_cairo_backend()

    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = Ferrite.getnnodes(mesh)

    # Resolve the topology vector. We don't pre-filter by density; the
    # per-cell alpha loop below applies `density_threshold` to render
    # below-threshold voxels as transparent.
    if topology === undef
        topology = ones(T, Ferrite.getncells(mesh))
    end

    # Auto-scale arrow sizes/linewidths/sliders from the mesh
    # characteristic length so visualizations look proportional
    # regardless of model scale.
    _L = _mesh_scale(mesh.nodes, dim)
    _auto = _auto_arrow_params(_L, dim)
    vector_arrowsize = vector_arrowsize === Auto ? _auto.arrow_size : vector_arrowsize
    load_arrow_linewidth =
        load_arrow_linewidth === Auto ? _auto.arrow_linewidth : load_arrow_linewidth
    support_arrow_linewidth =
        support_arrow_linewidth === Auto ? _auto.arrow_linewidth : support_arrow_linewidth
    arrow_quality = arrow_quality === Auto ? _auto.arrow_quality : arrow_quality
    default_support_scale =
        default_support_scale === Auto ? _auto.default_scale : default_support_scale
    default_load_scale =
        default_load_scale === Auto ? _auto.default_scale : default_load_scale
    scale_range = scale_range === Auto ? _auto.scale_range : scale_range
    default_exagg_scale = default_exagg_scale === Auto ? 1.0 : default_exagg_scale

    given_u = u !== undef
    cloaddict = cloaddict === undef ? getcloaddict(problem) : cloaddict

    # Cairo crashes cryptically on NaN linewidths/coordinates; fail fast with
    # the actual cause (e.g. a failed optimization returning a NaN minimizer)
    if topology !== undef
        all(isfinite, topology) || throw(
            ArgumentError(
                "visualize: topology contains non-finite values ($(count(!isfinite, topology)) of $(length(topology))) — the optimization likely failed",
            ),
        )
    end
    if given_u
        all(isfinite, u) || throw(
            ArgumentError(
                "visualize: u contains non-finite values ($(count(!isfinite, u)) of $(length(u))) — the FEA solve likely failed (e.g. singular stiffness matrix)",
            ),
        )
    end

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
    fig = Figure(; size=dim == 3 ? (1020, 510) : (800, 600))

    if dim == 2
        ax1 = Axis(fig[1, 1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        # https://jkrumbiegel.github.io/MakieLayout.jl/v0.3/layoutables/#LScene-1
        # https://makie.juliaplots.org/stable/cameras.html#D-Camera
        # ax1 = layout[1, 1] = LScene(scene, camera = cam3d!, raw = false)
        ax1 = LScene(fig[1, 1]; scenekw=(camera=(cam3d!), raw=false)) # , height=750
    end

    # * support / load appearance / deformation exaggeration control
    # Sliders only exist in interactive mode (and the CairoMakie backend
    # auto-disables `interactive` above).
    condition_lsgrid = nothing
    deform_lsgrid = nothing
    if display_supports && interactive
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
    if given_u && interactive
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

    dup_nodes, dup_cells, _, old_node_id_from_new = _explode_nodes_and_cells(mesh)
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
        # Cells below the density threshold draw with zero alpha — they
        # are still in the mesh so the topology is preserved, but
        # invisible in the rendering.
        if cell_xvar < density_threshold
            alpha = 0.0
        elseif alpha_from_density
            alpha = cell_xvar
        else
            alpha = one(cell_xvar)
        end
        for new_nid in dup_cells[i].nodes
            ccolor = undeformed_mesh_color
            if cell_colors !== undef
                ccolor = ColorSchemes.get(colormap, scaled_cell_colors[i])
            end
            undeformed_mesh_colors[new_nid] = RGBAf(ccolor.r, ccolor.g, ccolor.b, alpha)
        end
    end
    # * Procedural surface texture overlay (independent of cell density and
    # cell_colors; cosmetic only). `:none` is a no-op.
    if surface_texture !== :none
        bbox_lo, bbox_hi = boundingbox(problem.ch.dh.grid)
        period = if texture_period === Auto
            _auto_texture_period(bbox_lo, bbox_hi)
        else
            Float64(texture_period)
        end
        bbox_lo3 = if dim == 2
            (Float64(bbox_lo[1]), Float64(bbox_lo[2]), 0.0)
        else
            (Float64.(bbox_lo)...,)
        end
        bbox_hi3 = if dim == 2
            (Float64(bbox_hi[1]), Float64(bbox_hi[2]), 0.0)
        else
            (Float64.(bbox_hi)...,)
        end
        _apply_surface_texture!(
            undeformed_mesh_colors,
            dup_nodes,
            surface_texture,
            period,
            texture_angle,
            texture_color,
            texture_strength,
            bbox_lo3,
            bbox_hi3,
        )
    end
    if cell_colors !== undef && draw_legend
        _create_colorbar(fig[1, 2], colormap, cell_colors)
    end

    # * Undeformed mesh
    _setup_lighting!(ax1, lighting)
    mesh_kwargs = (; shading=_shading_for(lighting), kw...)
    Makie.mesh!(
        ax1,
        dup_nodes,
        dup_cells;
        color=undeformed_mesh_colors,
        specular=0.4f0,
        shininess=32.0f0,
        mesh_kwargs...,
    )
    dim == 3 && _plot_voxel_edges!(ax1, dup_nodes, dup_cells, undeformed_mesh_colors)

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

        deformation_scale =
            interactive ? deform_lsgrid.sliders[1].value : Observable(default_exagg_scale)
        exagg_deformed_nodes = lift(
            s -> [
                Ferrite.Node(
                    Tuple([new_node.x[ax_id] + s * dup_u[ax_id, nid] for ax_id in 1:3])
                ) for (nid, new_node) in enumerate(dup_nodes)
            ],
            deformation_scale,
        )
        deformed_mesh_colors = [
            RGBAf(
                deformed_mesh_color.r,
                deformed_mesh_color.g,
                deformed_mesh_color.b,
                ccolor.alpha,
            ) for ccolor in undeformed_mesh_colors
        ]
        Makie.mesh!(
            ax1,
            exagg_deformed_nodes,
            dup_cells;
            color=deformed_mesh_colors,
            shading=_shading_for(lighting),
            specular=0.4f0,
            shininess=32.0f0,
        )
    end

    if display_supports
        # TODO pressure loads?
        # * load vectors
        if cloaddict !== undef
            if length(cloaddict) > 0
                loaded_nodes = Point3f.(nodes[node_ind].x for (node_ind, _) in cloaddict)
                load_items = collect(cloaddict)
                loaded_nodes = Point3f.(nodes[node_ind].x for (node_ind, _) in load_items)

                # `live_load_scale` raises the load arrow magnitude over time;
                # without it (interactive=false) we use the static default.
                live_load_scale =
                    interactive ? condition_lsgrid.sliders[2].value : default_load_scale
                load_dirs = lift(live_load_scale) do s
                    if dim == 2
                        [Vec2f(s * lv[1], s * lv[2]) for (_, lv) in load_items]
                    else
                        [Vec3f(s * lv[1], s * lv[2], s * lv[3]) for (_, lv) in load_items]
                    end
                end

                _plot_arrows!(
                    ax1,
                    loaded_nodes,
                    load_dirs;
                    arrow_color=load_arrow_color,
                    arrow_linewidth=load_arrow_linewidth,
                    arrow_quality=arrow_quality,
                    arrow_size=vector_arrowsize,
                    dim=dim,
                )
            end
        end

        # * support vectors — one arrow per constrained direction at the
        # centroid of the constrained facets (instead of one arrow per node,
        # which becomes visually noisy on fine meshes). The constrained
        # nodes themselves are highlighted by a prominent scatter marker
        # with a white stroke so the user can identify which points are
        # actually constrained.
        ch = problem.ch
        live_support_scale =
            interactive ? condition_lsgrid.sliders[1].value : default_support_scale
        # Normalize the supports to (component, node_ids) pairs. `support_spec`
        # overrides the problem's Dirichlet BCs (used by the level-set viewers
        # whose equivalent problem has different supports).
        specs = if support_spec !== nothing
            support_spec
        else
            specs = Tuple{Int,Vector{Int}}[]
            for (_, dbc) in enumerate(ch.dbcs)
                node_ids = dbc.facets
                # Node-based BCs store node indices directly; facet-based BCs
                # (e.g. LBeam) store FacetIndex values — expand them to the
                # facet nodes so the supports can be drawn at node positions.
                support_ids = if eltype(node_ids) <: Ferrite.FacetIndex
                    unique(
                        Iterators.flatten(
                            Ferrite.facets(getcells(mesh, fi[1]))[fi[2]] for fi in node_ids
                        ),
                    )
                else
                    collect(node_ids)
                end
                for comp in dbc.components
                    push!(specs, (comp, support_ids))
                end
            end
            specs
        end
        drawn_scatter = Set{Vector{Int}}()
        for (comp, support_ids) in specs
            v = if comp == 1
                [1.0, 0.0, 0.0]
            elseif comp == 2
                [0.0, 1.0, 0.0]
            else
                [0.0, 0.0, 1.0]
            end
            fixed_pts = [Point3f(nodes[i].x...) for i in support_ids]
            centroid = Point3f(
                sum(p[1] for p in fixed_pts) / length(fixed_pts),
                sum(p[2] for p in fixed_pts) / length(fixed_pts),
                sum(p[3] for p in fixed_pts) / length(fixed_pts),
            )
            # One arrow per constrained direction; the constraints move
            # uniformly when the slider changes (live_support_scale).
            _plot_arrows!(
                ax1,
                [centroid],
                lift(live_support_scale) do s
                    return if dim == 2
                        [Vec2f(s * v[1], s * v[2])]
                    else
                        [Vec3f(s * v[1], s * v[2], s * v[3])]
                    end
                end;
                arrow_color=support_arrow_color,
                arrow_linewidth=0.75 * support_arrow_linewidth,
                arrow_quality=arrow_quality,
                arrow_size=0.65 * vector_arrowsize,
                dim=dim,
            )
            # BC markers: small, bright yellow with a thick black outline so
            # they stay readable against any mesh background but don't compete
            # with the support arrow for visual attention. Drawn once per node
            # set (several components can share the same nodes).
            if !(support_ids in drawn_scatter)
                push!(drawn_scatter, support_ids)
                Makie.scatter!(
                    ax1,
                    fixed_pts;
                    color=RGBAf(1.0, 0.9, 0.45, 1.0),
                    strokecolor=:black,
                    strokewidth=1.0,
                    markersize=3.5,
                )
            end
        end
    end # end if display_supports

    # * legend
    if draw_legend
        _add_legend!(
            fig,
            undeformed_mesh_color,
            load_arrow_color,
            support_arrow_color;
            has_colorbar=cell_colors !== undef,
            plot_height=dim == 3 ? 490 : 387,
        )
    end
    return fig
end

"""
    visualize(problem::TrussProblem{xdim,T};
        static=false,
        u=undef,
        topology=undef,
        undeformed_mesh_color=RGBAf(0, 0, 0, 1.0),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
        display_supports=true,
        vector_arrowsize=Auto,
        default_support_scale=Auto,
        default_load_scale=Auto,
        scale_range=1.0,
        default_exagg_scale=1.0,
        exagg_range=10.0,
        default_element_linewidth_scale=6.0,
        element_linewidth_range=10.0,
        kw...
    ) where {xdim,T}

Visualize a truss topology optimization problem. Loads and supports are drawn
as arrows, and the element line width and color can encode the design density.

# Inputs

- `problem`: truss topopt problem

# Optional arguments

- `static=false`: when `true`, return a `Bonito.App` with client-side camera controls instead of a `Makie.Figure`.
- `u=undef`: nodal displacement vector. If `undef`, assumed to be a zero vector.
- `topology=undef`: desired topology density vector. If `undef`, assume all cells are included.
- `undeformed_mesh_color`: color used for displaying the undeformed elements.
- `cell_colors=undef`: Vector of a value per cell to show the color map. If used, `undeformed_mesh_color` is ignored.
- `draw_legend=false`: draw the color legend for cell_colors.
- `colormap=ColorSchemes.Spectral_10`: color map used to show `cell_colors`.
- `deformed_mesh_color`: color used for displaying the deformed mesh if `u` is specified.
- `display_supports=true`: draw the support (Dirichlet BC) markers and arrows.
- `vector_arrowsize=Auto`: arrow size used for displaying loads and supports vectors.
- `default_support_scale=Auto`, `default_load_scale=Auto`: default support/load arrow scale.
- `scale_range=1.0`: upper limit of the sliders controlling the support and load scale sliders.
- `default_exagg_scale=1.0`: default deformation exaggeration scale.
- `exagg_range=10.0`: upper limit of the slider controlling the deformation exaggeration slider.
- `default_element_linewidth_scale=6.0`, `element_linewidth_range=10.0`: default and range of the element line-width slider.
- `kw...`: optional keyword argument passed to Makie.mesh! function.

# Returns
- `Makie.Figure` handle

"""
function TopOpt.visualize(
    problem::TrussProblem{xdim,T};
    static=false,
    u=undef,
    topology=undef,
    undeformed_mesh_color=RGBAf(0, 0, 0, 1.0),
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
    display_supports=true,
    vector_arrowsize=Auto,
    default_support_scale=Auto,
    default_load_scale=Auto,
    scale_range=1.0,
    default_exagg_scale=1.0,
    exagg_range=10.0,
    default_element_linewidth_scale=6.0,
    element_linewidth_range=10.0,
    kw...,
) where {xdim,T}
    if static
        return TopOpt._static_visualization(
            problem;
            u,
            topology,
            undeformed_mesh_color,
            cell_colors,
            draw_legend=true,
            colormap,
            deformed_mesh_color,
            display_supports,
            vector_arrowsize,
            default_support_scale,
            default_load_scale,
            scale_range,
            default_exagg_scale,
            exagg_range,
            default_element_linewidth_scale,
            element_linewidth_range,
            kw...,
        )
    end
    ndim = getdim(problem)
    ncells = Ferrite.getncells(problem)
    nnodes = Ferrite.getnnodes(problem)
    given_u = u !== undef
    topology = topology == undef ? ones(T, ncells) : topology

    # Cairo crashes cryptically on NaN linewidths; fail fast with the cause
    # (e.g. a failed optimization returning a NaN minimizer)
    all(isfinite, topology) || throw(
        ArgumentError(
            "visualize: topology contains non-finite values ($(count(!isfinite, topology)) of $(length(topology))) — the optimization likely failed",
        ),
    )
    if given_u
        all(isfinite, u) || throw(
            ArgumentError(
                "visualize: u contains non-finite values ($(count(!isfinite, u)) of $(length(u))) — the FEA solve likely failed",
            ),
        )
    end

    # Auto-scale arrow sizes and scales from the truss bounding-box diagonal
    # so loads and supports are a reasonable fraction of the structure.
    nodes = problem.truss_grid.grid.nodes
    _L = _mesh_scale(nodes, ndim)
    vector_arrowsize =
        vector_arrowsize === Auto ? clamp(0.08 * _L, 0.01, 100.0) : vector_arrowsize
    default_support_scale = default_support_scale === Auto ? 1.0 : default_support_scale
    default_load_scale = default_load_scale === Auto ? 1.0 : default_load_scale

    fig = Figure(; size=xdim == 3 ? (1020, 510) : (800, 600))
    if ndim == 2
        ax1 = Axis(fig[1, 1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        ax1 = LScene(fig[1, 1]; scenekw=(camera=(cam3d!), raw=false)) #, height=750)
    end

    # Fixed appearance values replace the interactive sliders in the static
    # export; the auto-scaled defaults above keep them proportional to the
    # structure size.
    linewidth_value = Observable(default_element_linewidth_scale)
    support_scale_value = Observable(default_support_scale)
    load_scale_value = Observable(default_load_scale)
    exagg_scale_value = Observable(default_exagg_scale)

    # * undeformed truss elements
    PtT = ndim == 2 ? Point2f : Point3f
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
    element_linewidth = lift(s -> topology_linewidth .* s, linewidth_value)
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
            exagg_scale_value,
        )
        linesegments!(
            ax1, exagg_edge_pts; linewidth=element_linewidth, color=deformed_mesh_color
        )
    end

    if display_supports
        # * load vectors
        loaded_nodes = [PtT(nodes[node_id].x) for node_id in keys(problem.force)]
        load_dirs = [PtT(force / norm(force)) for force in values(problem.force)]
        scaled_load_dirs = lift(s -> [dir * s for dir in load_dirs], load_scale_value)
        if ndim == 2
            dirs_obs = lift(dirs -> Vec2f.(dirs), scaled_load_dirs)
            Makie.arrows2d!(
                ax1, loaded_nodes, dirs_obs; color=:purple, lengthscale=vector_arrowsize
            )
        else
            dirs_obs = lift(dirs -> Vec3f.(dirs), scaled_load_dirs)
            Makie.arrows3d!(
                ax1, loaded_nodes, dirs_obs; color=:purple, lengthscale=vector_arrowsize
            )
        end

        Makie.scatter!(ax1, loaded_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))

        # * fixties vectors
        ch = problem.ch
        for (_, dbc) in enumerate(ch.dbcs)
            support_vectors = []
            node_ids = dbc.facets
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
                support_dir = [PtT(v) for _ in node_ids]
                if ndim == 2
                    Makie.arrows2d!(
                        ax1,
                        fixed_nodes,
                        lift(support_scale_value) do s
                            return [Vec2f(s * v[1], s * v[2]) for _ in node_ids]
                        end;
                        color=:orange,
                        lengthscale=vector_arrowsize,
                    )
                else
                    Makie.arrows3d!(
                        ax1,
                        fixed_nodes,
                        lift(support_scale_value) do s
                            return [Vec3f(s * v[1], s * v[2], s * v[3]) for _ in node_ids]
                        end;
                        color=:orange,
                        lengthscale=vector_arrowsize,
                    )
                end
            end
            Makie.scatter!(ax1, fixed_nodes) #, markersize = lift(s -> s * 3, lsgrid.sliders[1].value))
        end
    end # end if display_supports

    return fig
end

"""
    function visualize(problem::HeatTransferTopOptProblem{dim,T};
        static=false,
        topology=undef,
        undeformed_mesh_color=dim == 2 ? RGBAf(0.35, 0.35, 0.35, 1.0) :
            RGBAf(0.6, 0.6, 0.6, 0.85),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        surface_texture=:crosshatch,
        texture_period=Auto,
        texture_angle=45.0,
        texture_color=RGBAf(0.18, 0.26, 0.4, 1.0),
        texture_strength=0.06,
        density_threshold=0.5,
        alpha_from_density=true,
        lighting=:none,
        kw...
    ) where {dim,T}

Visualize a heat transfer topology optimization problem.

# Inputs

- `problem`: heat transfer topopt problem

# Optional arguments

- `static=false`: when `true`, return a `Bonito.App` with client-side camera controls instead of a `Makie.Figure`.
- `topology=undef`: desired topology density vector. If `undef`, assume all cells are included.
- `undeformed_mesh_color`: color used for displaying the mesh.
- `cell_colors=undef`: Vector of a value per cell to show the color map. Pass the
  per-cell temperature (e.g. `cell_temperature(TemperatureFun(solver)(x), problem)`)
  to color the elements by temperature.
- `draw_legend=false`: draw the color legend for cell_colors.
- `colormap=ColorSchemes.Spectral_10`: color map used to show `cell_color`.
- `lighting=:none`: lighting mode for the 3D scene (`:none` or `:default`).
- `kw...`: optional keyword argument passed to Makie.mesh! function.

# Returns
- `Makie.Figure` handle

"""
function TopOpt.visualize(
    problem::HeatTransferTopOptProblem{dim,T};
    static=false,
    topology=undef,
    undeformed_mesh_color=if dim == 2
        RGBAf(0.35, 0.35, 0.35, 1.0)
    else
        RGBAf(0.6, 0.6, 0.6, 0.85)
    end,
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    surface_texture=:crosshatch,
    texture_period=Auto,
    texture_angle=45.0,
    texture_color=RGBAf(0.18, 0.26, 0.4, 1.0),
    texture_strength=0.06,
    density_threshold=0.5,
    alpha_from_density=true,
    lighting=:none,
    kw...,
) where {dim,T}
    if static
        return TopOpt._static_visualization(
            problem;
            topology,
            undeformed_mesh_color,
            cell_colors,
            draw_legend=true,
            colormap,
            surface_texture,
            texture_period,
            texture_angle,
            texture_color,
            texture_strength,
            density_threshold,
            alpha_from_density,
            lighting,
            kw...,
        )
    end
    mesh = problem.ch.dh.grid
    nnodes = Ferrite.getnnodes(mesh)

    # Resolution guarantees the per-cell alpha loop below sees a Vector.
    if topology === undef
        topology = ones(T, Ferrite.getncells(mesh))
    end

    # Convert 2D nodes to 3D for visualization
    nodes = Vector{Ferrite.Node}(undef, nnodes)
    if dim == 2
        for (i, node) in enumerate(mesh.nodes)
            nodes[i] = Ferrite.Node((node.x[1], node.x[2], zero(T)))
        end
    else
        nodes = mesh.nodes
    end

    # Initialize Makie figure
    fig = Figure(; size=dim == 3 ? (1020, 510) : (800, 600))

    if dim == 2
        ax1 = Axis(fig[1, 1])
        ax1.aspect = DataAspect()
    else
        ax1 = LScene(fig[1, 1]; scenekw=(camera=(cam3d!), raw=false))
    end

    # Explode nodes and cells for per-cell coloring
    dup_nodes, dup_cells, _, old_node_id_from_new = _explode_nodes_and_cells(mesh)

    # Color per cell
    undeformed_mesh_colors = Vector{RGBAf}(undef, length(dup_nodes))
    scaled_cell_colors = similar(topology)
    scaled_cell_colors .= 0.0
    if cell_colors !== undef
        @assert length(cell_colors) == length(topology)
        val_range = maximum(cell_colors) - minimum(cell_colors)
        scaled_cell_colors = (cell_colors .- minimum(cell_colors)) / val_range
    end
    for i in eachindex(dup_cells)
        cell_xvar = topology[i]
        # Cells below `density_threshold` are rendered with zero alpha.
        # Avoids the user seeing half-baked optimization voxels.
        if cell_xvar < density_threshold
            alpha = 0.0
        elseif alpha_from_density
            alpha = cell_xvar
        else
            alpha = one(cell_xvar)
        end
        for new_nid in dup_cells[i].nodes
            ccolor = undeformed_mesh_color
            if cell_colors !== undef
                ccolor = ColorSchemes.get(colormap, scaled_cell_colors[i])
            end
            undeformed_mesh_colors[new_nid] = RGBAf(ccolor.r, ccolor.g, ccolor.b, alpha)
        end
    end
    if surface_texture !== :none
        bbox_lo, bbox_hi = boundingbox(problem.ch.dh.grid)
        period = if texture_period === Auto
            _auto_texture_period(bbox_lo, bbox_hi)
        else
            Float64(texture_period)
        end
        bbox_lo3 = if dim == 2
            (Float64(bbox_lo[1]), Float64(bbox_lo[2]), 0.0)
        else
            (Float64.(bbox_lo)...,)
        end
        bbox_hi3 = if dim == 2
            (Float64(bbox_hi[1]), Float64(bbox_hi[2]), 0.0)
        else
            (Float64.(bbox_hi)...,)
        end
        _apply_surface_texture!(
            undeformed_mesh_colors,
            dup_nodes,
            surface_texture,
            period,
            texture_angle,
            texture_color,
            texture_strength,
            bbox_lo3,
            bbox_hi3,
        )
    end
    if cell_colors !== undef && draw_legend
        _create_colorbar(fig[1, 2], colormap, cell_colors)
    end

    # Draw mesh: apply lighting, then plot. `:default` lighting adds
    # ambient + directional lights to the scene and switches shading to
    # FastShading with Phong-style specular/shininess.
    _setup_lighting!(ax1, lighting)
    Makie.mesh!(
        ax1,
        dup_nodes,
        dup_cells;
        color=undeformed_mesh_colors,
        shading=_shading_for(lighting),
        specular=0.4f0,
        shininess=32.0f0,
    )
    dim == 3 && _plot_voxel_edges!(ax1, dup_nodes, dup_cells, undeformed_mesh_colors)
    return fig
end

"""
    visualize(result::OpenLSTO.LevelSetResult; ...)

Visualize the result of `OpenLSTO.compliance_minimization` with the regular
continuum visualizer. The level-set design is converted to a
`PointLoadCantilever` problem on the same grid and the per-cell area
fractions are shown as the density field, so the same controls (loads,
supports, sliders, `static=true` viewer) apply as for SIMP results.

See [`visualize(::StiffnessTopOptProblem)`](@ref) for the shared keyword
arguments. Extra keywords:
- `E`, `ν`, `force`: material/load parameters of the equivalent
  `PointLoadCantilever` (only used for drawing loads and supports; defaults
  match `compliance_minimization`).

The load and support arrows are taken from the result's own
`boundary_conditions`, so an L-beam result draws its top-edge supports and
2/5-height load rather than the equivalent cantilever's.
"""
function TopOpt.visualize(
    result::TopOpt.OpenLSTO.LevelSetResult;
    static=false,
    topology=nothing,
    E=1.0,
    ν=0.3,
    force=0.5,
    kw...,
)
    mesh = result.study.mesh
    problem = PointLoadCantilever((mesh.nelx, mesh.nely), (1.0, 1.0), E, ν, force)
    topology = topology === nothing ? TopOpt.OpenLSTO.area_fractions(result) : topology
    bc = result.boundary_conditions
    cloaddict = Dict(node => v for (node, v) in bc.loads)
    return TopOpt.visualize(
        problem;
        static=static,
        topology=topology,
        cloaddict=cloaddict,
        support_spec=bc.supports,
        kw...,
    )
end

"""
    visualize(level_set::OpenLSTO.LevelSet3D; ...)

Visualize a 3D level-set design with the regular 3D continuum visualizer. The
per-cell volume fractions are shown as the density field of an equivalent
`PointLoadCantilever` problem, so the same controls (loads, supports, sliders,
`static=true` viewer) apply as for SIMP results.

See [`visualize(::StiffnessTopOptProblem)`](@ref) for the shared keyword
arguments. Extra keywords:
- `E`, `ν`, `force`: material/load parameters of the equivalent
  `PointLoadCantilever` (only used for drawing loads and supports).

The load and support arrows are taken from the level set's own
`boundary_conditions` when available (set by `compliance_minimization_3d`),
otherwise the equivalent `PointLoadCantilever`'s. The equivalent problem
places its point load at the y-z midpoint, so `ny` and `nz` must be even.
"""
function TopOpt.visualize(
    level_set::TopOpt.OpenLSTO.LevelSet3D;
    static=false,
    topology=nothing,
    E=1.0,
    ν=0.3,
    force=1.0,
    kw...,
)
    if length(level_set.volumefraction_vector) != level_set.num_cells
        TopOpt.OpenLSTO.calculate_volume_fractions!(level_set)
    end
    problem = PointLoadCantilever(
        (level_set.nx, level_set.ny, level_set.nz), (1.0, 1.0, 1.0), E, ν, force
    )
    topology = topology === nothing ? level_set.volumefraction_vector : topology
    bc = level_set.boundary_conditions
    if bc === nothing
        return TopOpt.visualize(problem; static=static, topology=topology, kw...)
    end
    cloaddict = Dict(node => v for (node, v) in bc.loads)
    return TopOpt.visualize(
        problem;
        static=static,
        topology=topology,
        cloaddict=cloaddict,
        support_spec=bc.supports,
        kw...,
    )
end

include("static_viewer.jl")
end
