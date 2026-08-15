module TopOptMakieExt

using LinearAlgebra: norm
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
    HeatConductionProblem
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
        return min(
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
            push!(scene.lights, Makie.AmbientLight(RGBf(0.55, 0.55, 0.6)))
        end
        if !any(l -> l isa Makie.DirectionalLight, scene.lights)
            push!(
                scene.lights,
                Makie.DirectionalLight(RGBf(0.9, 0.9, 0.85), Vec3f(-0.4, -0.5, -0.7)),
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
slightly translucent lighter gray for 3D. Used by `visualize_static` to
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
    arrow_size = clamp(0.05 * L, 0.01, 100.0)
    arrow_linewidth = dim == 2 ? 4.0 : max(arrow_size / 20, 0.05)
    default_scale = clamp(1.0, 0.001, 1000.0)
    scale_range = clamp(20.0 * default_scale, 0.1, 10000.0)
    arrow_quality = 20
    return (; arrow_size, arrow_linewidth, default_scale, scale_range, arrow_quality)
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
        )
    end
    Makie.scatter!(ax, points; color=arrow_color, markersize=4)
    return nothing
end

"""
Add a legend identifying the undeformed mesh, deformed mesh, load
arrows, and support arrows. Skipped on the CairoMakie backend (Legend
blocks render inconsistently in PDF/SVG exports).
"""
function _add_legend!(
    fig,
    undeformed_mesh_color,
    deformed_mesh_color,
    load_arrow_color,
    support_arrow_color;
    has_colorbar::Bool=false,
)
    _is_cairo_backend() && return fig
    elements = [
        Makie.PolyElement(; color=undeformed_mesh_color),
        Makie.PolyElement(; color=deformed_mesh_color),
        Makie.MarkerElement(; marker=:circle, color=load_arrow_color),
        Makie.MarkerElement(; marker=:circle, color=support_arrow_color),
    ]
    labels = ["undeformed mesh", "deformed mesh", "load", "support"]
    Makie.Legend(
        fig[1, 1],
        elements,
        labels;
        tellheight=false,
        tellwidth=false,
        framevisible=true,
        backgroundcolor=RGBAf(1, 1, 1, 0.85),
        position=(1, 1),
    )
    return fig
end

################################

"""
    function visualize(problem::StiffnessTopOptProblem{dim,T};
        u=undef,
        topology=undef,
        cloaddict=undef,
        undeformed_mesh_color=dim == 2 ? RGBAf(0.35, 0.35, 0.35, 1.0) :
            RGBAf(0.6, 0.6, 0.6, 0.85),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
        surface_texture=:none,
        texture_period=Auto,
        texture_angle=45.0,
        texture_color=RGBAf(0.0, 0.0, 0.0, 1.0),
        texture_strength=0.2,
        density_threshold=0.5,
        alpha_from_density=true,
        interactive=true,
        vector_arrowsize=Auto,
        # Red loads and blue supports remain legible without overpowering the
        # density colormap or lighting.
        load_arrow_color=RGBAf(0.85, 0.12, 0.08, 1.0),
        support_arrow_color=RGBAf(0.08, 0.28, 0.75, 1.0),
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
- `vector_arrowsize=10.0` : the vector arrow size used for displaying loads and supports vectors.- `default_support_scale=1.0` : the default support scale used in the slider.
- `default_load_scale=1.0` : the default load scale used in the slider.
- `scale_range=1.0` : the upper limit of the sliders controlling the support and load scale sliders.
- `default_exagg_scale=1.0` : default deformation exaggeration scale.
- `exagg_range=10.0` : the upper limit of the slider controlling the deformation exaggeration slider.
- `kw...` : optional keyword argument passed to [Makie.mesh!](https://docs.makie.org/stable/api/#mesh!) function.
- `static=false` : when `true`, return a `Bonito.App` with client-side camera controls instead of a `Makie.Figure`.

# Returns
- `Makie.Figure` handle

"""
function TopOpt.visualize(
    problem::StiffnessTopOptProblem{dim,T};
    static=false,
    u=undef,
    topology=undef,
    cloaddict=undef,
    undeformed_mesh_color=if dim == 2
        RGBAf(0.35, 0.35, 0.35, 1.0)
    else
        RGBAf(0.6, 0.6, 0.6, 0.85)
    end,
    cell_colors=undef,
    draw_legend=false,
    colormap=ColorSchemes.Spectral_10,
    deformed_mesh_color=RGBAf(0, 1, 1, 0.4),
    surface_texture=:none,
    texture_period=Auto,
    texture_angle=45.0,
    texture_color=RGBAf(0.0, 0.0, 0.0, 1.0),
    texture_strength=0.4,
    density_threshold=0.5,
    alpha_from_density=true,
    interactive=true,
    display_supports=true,
    lighting=:default,
    vector_arrowsize=Auto,
    load_arrow_color=RGBAf(0.85, 0.12, 0.08, 1.0),
    support_arrow_color=RGBAf(0.08, 0.28, 0.75, 1.0),
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
        return TopOpt.visualize_static(
            problem;
            u,
            topology,
            cloaddict,
            undeformed_mesh_color,
            cell_colors,
            draw_legend,
            colormap,
            deformed_mesh_color,
            surface_texture,
            texture_period,
            texture_angle,
            texture_color,
            texture_strength,
            density_threshold,
            alpha_from_density,
            interactive,
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
        _apply_surface_texture!(
            undeformed_mesh_colors,
            dup_nodes,
            surface_texture,
            period,
            texture_angle,
            texture_color,
            texture_strength,
            (Float64.(bbox_lo)...,),
            (Float64.(bbox_hi)...,),
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
        for (_, dbc) in enumerate(ch.dbcs)
            support_vectors = Tuple{Int,Vector{Float64}}[]
            if 1 in dbc.components
                push!(support_vectors, (1, [1.0, 0.0, 0.0]))
            end
            if 2 in dbc.components
                push!(support_vectors, (2, [0.0, 1.0, 0.0]))
            end
            if 3 in dbc.components
                push!(support_vectors, (3, [0.0, 0.0, 1.0]))
            end
            node_ids = dbc.facets
            # Node-based BCs store node indices directly; facet-based BCs
            # (e.g. LBeam) store FacetIndex values — expand them to the facet
            # nodes so the supports can be drawn at node positions.
            support_ids = if eltype(node_ids) <: Ferrite.FacetIndex
                unique(
                    Iterators.flatten(
                        Ferrite.facets(getcells(mesh, fi[1]))[fi[2]] for fi in node_ids
                    ),
                )
            else
                collect(node_ids)
            end
            fixed_pts = [Point3f(nodes[i].x...) for i in support_ids]
            centroid = Point3f(
                sum(p[1] for p in fixed_pts) / length(fixed_pts),
                sum(p[2] for p in fixed_pts) / length(fixed_pts),
                sum(p[3] for p in fixed_pts) / length(fixed_pts),
            )
            # One arrow per constrained direction; the constraints move
            # uniformly when the slider changes (live_support_scale).
            for (_comp, v) in support_vectors
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
                    arrow_linewidth=support_arrow_linewidth,
                    arrow_quality=arrow_quality,
                    arrow_size=vector_arrowsize,
                    dim=dim,
                )
            end
            # BC markers: small, bright yellow with a thick black outline so
            # they stay readable against any mesh background but don't
            # compete with the support arrow for visual attention.
            Makie.scatter!(
                ax1,
                fixed_pts;
                color=support_arrow_color,
                strokecolor=:black,
                strokewidth=2.0,
                markersize=5,
            )
        end
    end # end if display_supports

    # * legend
    if draw_legend
        _add_legend!(
            fig,
            undeformed_mesh_color,
            deformed_mesh_color,
            load_arrow_color,
            support_arrow_color;
            has_colorbar=cell_colors !== undef,
        )
    end

    return fig
end

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
    vector_arrowsize=10.0,
    default_support_scale=1e-2,
    default_load_scale=1e-2,
    scale_range=1.0,
    default_exagg_scale=1.0,
    exagg_range=10.0,
    default_element_linewidth_scale=6.0,
    element_linewidth_range=10.0,
    kw...,
) where {xdim,T}
    if static
        return TopOpt.visualize_static(
            problem;
            u,
            topology,
            undeformed_mesh_color,
            cell_colors,
            draw_legend,
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

    fig = Figure()
    if ndim == 2
        ax1 = Axis(fig[1, 1])
        # tightlimits!(ax1)
        # ax1.aspect = AxisAspect(1)
        ax1.aspect = DataAspect()
    else
        ax1 = LScene(fig[1, 1]; scenekw=(camera=(cam3d!), raw=false)) #, height=750)
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
        load_dirs = [PtT(force / norm(force)) for force in values(problem.force)]
        scaled_load_dirs = lift(
            s -> [dir * s for dir in load_dirs], condition_lsgrid.sliders[2].value
        )
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
                        lift(condition_lsgrid.sliders[1].value) do s
                            return [Vec2f(s * v[1], s * v[2]) for _ in node_ids]
                        end;
                        color=:orange,
                        lengthscale=vector_arrowsize,
                    )
                else
                    Makie.arrows3d!(
                        ax1,
                        fixed_nodes,
                        lift(condition_lsgrid.sliders[1].value) do s
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
        topology=undef,
        undeformed_mesh_color=dim==2 ? RGBAf(0,0,0,1.0) : RGBAf(0.5,0.5,0.5,0.4),
        cell_colors=undef,
        draw_legend=false,
        colormap=ColorSchemes.Spectral_10,
        kw...
    ) where {dim,T}

Visualize a heat transfer topology optimization problem.

# Inputs

- `problem`: heat transfer topopt problem

# Optional arguments

- `topology=undef`: desired topology density vector. If `undef`, assume all cells are included.
- `undeformed_mesh_color`: color used for displaying the mesh.
- `cell_colors=undef`: Vector of a value per cell to show the color map.
- `draw_legend=false`: draw the color legend for cell_colors.
- `colormap=ColorSchemes.Spectral_10`: color map used to show `cell_color`.
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
    surface_texture=:none,
    texture_period=Auto,
    texture_angle=45.0,
    texture_color=RGBAf(0.0, 0.0, 0.0, 1.0),
    texture_strength=0.4,
    density_threshold=0.5,
    alpha_from_density=true,
    lighting=:default,
    kw...,
) where {dim,T}
    if static
        return TopOpt.visualize_static(
            problem;
            topology,
            undeformed_mesh_color,
            cell_colors,
            draw_legend,
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
    fig = Figure()

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
        _apply_surface_texture!(
            undeformed_mesh_colors,
            dup_nodes,
            surface_texture,
            period,
            texture_angle,
            texture_color,
            texture_strength,
            (Float64.(bbox_lo)...,),
            (Float64.(bbox_hi)...,),
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

    return fig
end

# -- visualize_static --------------------------------------------------------

# Handle to the figure of the most recently built `visualize_static` app.
# The figure lives inside the Bonito session handler and is otherwise
# unreachable from Julia; this enables inspection of a live-served app
# (used by tests to verify the JS->Julia camera command channel).
const _last_static_fig = Ref{Any}(nothing)

"""
    visualize_static(problem; kwargs...)

Build a `Bonito.App` around `visualize(problem; interactive=false, kwargs...)`
with a full camera-control UI implemented in client-side JavaScript, so it
keeps working in statically exported HTML with no running Julia process
(e.g. Quarto/Documenter pages).

Controls mirror the live (Julia-backed) UI:
- Reset, Recenter
- editable camera fields (azimuth φ, elevation θ in degrees; eye x/y/z),
  kept in sync with mouse-driven camera motion
- view presets (Iso / Front / Back / Left / Right / Top / Bottom)
- orthographic toggle (approximated with a 1° telephoto perspective — see
  the implementation note in the JS bridge below)
- "Zoom to cursor" toggle: when ON, wheel/pinch zoom is anchored to the
  pointer position; button zoom always targets the lookat (center).
- Pan/zoom cross to the right of the figure as its own column.
- Save: downloads the WebGL canvas as a PNG with an editable filename

The whole app is centered in the browser viewport (both axes). Requires the
WGLMakie backend (`using WGLMakie`). For a self-contained page, call
`Bonito.Page(exportable=true, offline=true)` first.
"""
function TopOpt.visualize_static(
    problem::AbstractTopOptProblem;
    # Pulled out of kwargs so the in-app legend swatches can reference
    # them — matching the Stiffness visualize defaults. Passing them on
    # to `visualize` keeps the rendered colors the same as without
    # `visualize_static`.
    undeformed_mesh_color=_default_und_mesh_color(problem),
    load_arrow_color=RGBAf(0.85, 0.12, 0.08, 1.0),
    support_arrow_color=RGBAf(0.08, 0.28, 0.75, 1.0),
    lighting=:default,
    kw...,
)
    backend = current_backend()
    occursin("WGL", string(backend)) || throw(
        ArgumentError(
            "visualize_static requires the WGLMakie backend (found $(backend)); load it with `using WGLMakie`",
        ),
    )
    WGLMakie = backend
    Bonito = WGLMakie.Bonito
    D = Bonito.DOM
    Bonito.Page(; exportable=true, offline=true)
    # The js"..." macro resolves its module at parse time, which is impossible
    # here (Bonito is only reachable at runtime through the backend module),
    # so the JS is built as raw JSCode with values inlined as JS literals.
    jsvec(v) = "[" * join(Float64.(v), ",") * "]"

    return Bonito.App() do session
        fig = TopOpt.visualize(
            problem;
            interactive=false,
            undeformed_mesh_color=undeformed_mesh_color,
            load_arrow_color=load_arrow_color,
            support_arrow_color=support_arrow_color,
            lighting=lighting,
            kw...,
        )
        _last_static_fig[] = fig
        ax1_candidates = [c for c in fig.content if c isa LScene]
        # 2D problems use an Axis, not an LScene; there is no 3D camera to
        # control, so return the bare figure.
        isempty(ax1_candidates) && return D.div(fig)
        ax1 = first(ax1_candidates)

        scene_id = WGLMakie.js_uuid(ax1.scene)

        cam = ax1.scene.camera_controls
        lookat = Float64.(cam.lookat[])
        persp_fov = Float64(cam.fov[])

        # Compact styles sized for embedding contexts (Quarto's content
        # column, VSCode preview): small fonts, tight gaps, flex-wrap rows.
        btn = "font-size:11px;padding:2px 7px;margin:0;cursor:pointer;"
        # `num` width fits an optional sign + 3 integer digits + 1 decimal:
        # angles span −180..180 ("−180.0") and positions can reach ±999.9.
        num = "font-size:11px;width:7em;padding:1px 2px;"
        lab = "font-size:11px;user-select:none;cursor:pointer;white-space:nowrap;"
        row = "display:flex;flex-wrap:wrap;gap:3px;align-items:center;margin:2px 0;"
        cross_btn = btn * "width:24px;"

        # Bounds for the eye-position fields: generous but finite, so typing
        # or spinning cannot fling the camera to numerically absurd places.
        # Reset computes an isometric default view from the figure's bounding
        # box (centroid + true isometric 35.264° elevation / 45° azimuth).
        # This puts the model free of clipping from any facing direction
        # and lines up with the printed engineering convention; Reset and
        # the JS camera fields both return to this pose (+ one zoom-out
        # step for headroom).
        bbox_lo, bbox_hi = boundingbox(problem.ch.dh.grid)
        center = (
            Float64(bbox_hi[1] + bbox_lo[1]) / 2,
            Float64(bbox_hi[2] + bbox_lo[2]) / 2,
            Float64(bbox_hi[3] + bbox_lo[3]) / 2,
        )
        span = (
            Float64(bbox_hi[1] - bbox_lo[1]),
            Float64(bbox_hi[2] - bbox_lo[2]),
            Float64(bbox_hi[3] - bbox_lo[3]),
        )
        max_span = max(max(span[1], span[2]), span[3], 1.0)
        # True isometric: elevation atan(1/√2) ≈ 35.264° and azimuth 45°
        # put the eye equally down all three principal axes (the three eye
        # components have the same magnitude). This is the canonical
        # engineering / textbook isometric view.
        el = atand(1 / sqrt(2))
        az = 45.0
        dir_xyz = (cosd(el) * cosd(az), cosd(el) * sind(az), sind(el))
        eye_dist = 1.7 * max_span
        lookat = center

        # Reset defaults to one `−`-button step further out than the
        # canonical framing: leaves a margin of empty space around the
        # model and matches what the zoom-out button does, so clicking
        # either lands the eye in the same place.
        zoom_step_out = 1.125
        eye0 = (
            center[1] + dir_xyz[1] * eye_dist * zoom_step_out,
            center[2] + dir_xyz[2] * eye_dist * zoom_step_out,
            center[3] + dir_xyz[3] * eye_dist * zoom_step_out,
        )
        cam.lookat[] = Vec3f(lookat...)
        cam.eyeposition[] = Vec3f(eye0...)
        reach = 20 * eye_dist * zoom_step_out
        function coord_input(class, i)
            return D.input(;
                type="number",
                class,
                style=num,
                min=round(lookat[i] - reach; digits=1),
                max=round(lookat[i] + reach; digits=1),
                step="1",
            )
        end

        button(label, class) = D.button(label; class, style=btn)
        controls = D.div(
            D.div(
                button("Reset", "tv-reset"),
                button("Recenter", "tv-recenter"),
                D.span("Camera"; style=lab * "margin-left:8px;"),
                D.span("φ"; style=lab),
                D.input(;
                    type="number", class="tv-phi", style=num, min=-180, max=180, step="1"
                ),
                D.span("θ"; style=lab),
                D.input(;
                    type="number",
                    class="tv-theta",
                    style=num,
                    min=-89.9,
                    max=89.9,
                    step="1",
                ),
                D.span("x"; style=lab),
                coord_input("tv-x", 1),
                D.span("y"; style=lab),
                coord_input("tv-y", 2),
                D.span("z"; style=lab),
                coord_input("tv-z", 3);
                style=row,
            ),
            D.div(
                (
                    button(p, "tv-preset-$p") for
                    p in ("Iso", "Front", "Back", "Left", "Right", "Top", "Bottom")
                )...,
                D.label(
                    D.input(; type="checkbox", class="tv-ortho"),
                    " Orthographic";
                    style=lab * "margin-left:8px;",
                ),
                D.label(
                    D.input(; type="checkbox", checked=false, class="tv-zoomcursor"),
                    " Zoom to cursor";
                    style=lab,
                ),
                D.input(;
                    type="text",
                    value="topopt_view.png",
                    class="tv-savename",
                    style="font-size:11px;width:10em;padding:1px 3px;margin-left:8px;",
                ),
                button("Save", "tv-save");
                style=row,
            ),
        )

        # Zoom/pan cross anchored to the **3D viewport** (the LScene canvas),
        # not the browser viewport — so the buttons ride along with the
        # figure inside its column instead of floating to the page edges.
        # Compact: tight padding, small radius.
        cross = D.div(
            D.div(
                button("−", "tv-zoomout"),
                button("+", "tv-zoomin");
                style="display:flex;gap:2px;justify-content:center;",
            ),
            D.div(
                D.button("↑"; class="tv-panup", style=cross_btn);
                style="display:flex;justify-content:center;margin-top:2px;",
            ),
            D.div(
                D.button("←"; class="tv-panleft", style=cross_btn),
                D.button("→"; class="tv-panright", style=cross_btn);
                style="display:flex;gap:24px;justify-content:center;",
            ),
            D.div(
                D.button("↓"; class="tv-pandown", style=cross_btn);
                style="display:flex;justify-content:center;",
            );
            style=join([
                "position:absolute;",
                "bottom:6px;",
                "right:6px;",
                "z-index:20;",
                "background:rgba(255,255,255,0.85);",
                "padding:3px 5px;",
                "border-radius:3px;",
                "box-shadow:0 1px 3px rgba(0,0,0,0.15);",
            ]),
        )

        # No legend in the Bonito app — the design / load / support swatches
        # were dropped at the user's request, leaving just the controls
        # above and the figure (with the bottom-right zoom/pan cross) below.

        # Figure wrapper: `position:relative` so the absolutely-positioned
        # cross overlays the rendered 3D viewport (the LScene canvas) rather
        # than the browser chrome.
        figure = D.div(
            fig,
            cross;
            style=join([
                "position:relative;",
                "min-width:0;",
                "flex-shrink:1;",
                "display:inline-block;",
                "line-height:0;",  # collapse any baseline gap from the canvas
            ]),
        )

        container = D.div(
            # Compact layout: controls on top, figure below. The cross
            # sits inside the figure wrapper as `position:absolute` so it
            # rides along with the canvas when the figure is resized.
            controls,
            figure;
            style=join([
                "position:fixed;",
                "top:0;left:0;right:0;bottom:0;",
                "display:flex;",
                "flex-direction:column;",
                "justify-content:center;",
                "align-items:center;",
                "gap:8px;",                 # tighter vertical pitch
                "padding:8px;",             # tighter padding
                "box-sizing:border-box;",
                "overflow:auto;",
            ]),
        )

        # --- live/offline camera bridge -----------------------------------
        # In a static export the JS THREE.OrbitControls is the camera
        # authority. In a *live* session (VSCode/Quarto inline, browser with
        # a running Julia process) WGLMakie disables the JS OrbitControls
        # (its update() no-ops while `Bonito.can_send_to_julia()`), and
        # Julia's Camera3D is the authority instead. Every control therefore
        # branches: offline → drive OrbitControls; live → send a command
        # through `cam_cmd`, applied here via `update_cam!`. `cam_state`
        # mirrors the Julia camera back to JS so the readout fields track
        # mouse motion and so JS computes new poses from fresh state.
        #
        # cam_cmd:  [1, eye..., target..., up...]  set pose (up=0,0,0 keeps up)
        #           [2, flag]                      orthographic on/off
        # cam_state: [eye..., target..., up..., fov, ortho_flag]
        function state_vec()
            return Float64[
                cam.eyeposition[]...,
                cam.lookat[]...,
                cam.upvector[]...,
                cam.fov[],
                cam.settings.projectiontype[] == Makie.Orthographic ? 1.0 : 0.0,
            ]
        end
        cam_cmd = Bonito.Observable(Float64[])
        cam_state = Bonito.Observable(state_vec())
        sync_state!(_...) = (cam_state[] = state_vec(); nothing)
        on(cam_cmd) do v
            isempty(v) && return nothing
            op = Int(v[1])
            if op == 1
                length(v) == 10 || throw(
                    ArgumentError("camera pose command needs 10 values, got $(length(v))"),
                )
                eye = Vec3f(v[2], v[3], v[4])
                tgt = Vec3f(v[5], v[6], v[7])
                up = Vec3f(v[8], v[9], v[10])
                if norm(up) < 0.5
                    update_cam!(ax1.scene, cam, eye, tgt)
                else
                    update_cam!(ax1.scene, cam, eye, tgt, up)
                end
            elseif op == 2
                checked = v[2] > 0.5
                is_ortho = cam.settings.projectiontype[] == Makie.Orthographic
                if checked != is_ortho
                    # Same distance rescale as the live UI toggle: ortho's
                    # visible half-height at distance d is d itself,
                    # perspective's is d*tand(fov/2).
                    fov_scale = tand(0.5 * cam.fov[])
                    eyev = Vec3f(cam.eyeposition[])
                    lav = Vec3f(cam.lookat[])
                    dirv = eyev - lav
                    d = norm(dirv)
                    nd = checked ? d * fov_scale : d / fov_scale
                    cam.eyeposition[] = lav + (dirv / d) * nd
                    cam.settings.projectiontype[] =
                        checked ? Makie.Orthographic : Makie.Perspective
                end
            else
                throw(ArgumentError("unknown camera command opcode $op"))
            end
            sync_state!()
            return nothing
        end
        # Mirror mouse-driven camera motion (throttled: each sync rewrites
        # five textboxes in JS).
        on(sync_state!, throttle(0.3, cam.eyeposition))
        on(sync_state!, throttle(0.3, cam.lookat))

        # All behavior is wired in one onload script: query controls by
        # class, poll until WGLMakie has registered the scene (the WebGL
        # canvas initializes asynchronously), then attach listeners.
        setup_js = """
            const q = (cls) => container.querySelector('.' + cls);
            const perspFov = $(persp_fov);
            const la0 = $(jsvec(lookat));
            const eye0 = $(jsvec(eye0));
            const deg = Math.PI / 180;
            // Offline orthographic approximation: WGLMakie's JS camera sync
            // is a closure over a PerspectiveCamera, so a true
            // OrthographicCamera cannot be swapped in; a 1° telephoto is
            // visually indistinguishable from parallel projection. Live
            // sessions use Makie's real Orthographic instead (op 2).
            const ORTHO_FOV = 1.0;
            const live = () => Bonito.can_send_to_julia && Bonito.can_send_to_julia();

            const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
            const len = (a) => Math.hypot(a[0], a[1], a[2]);
            const cross = (a, b) => [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            const normv = (a) => scl(a, 1 / len(a));

            const poll = setInterval(() => {
                const scene = window.WGL.find_scene('$(scene_id)');
                if (!scene || !scene.orbitcontrols) return;
                clearInterval(poll);
                const c = scene.orbitcontrols;

                // Current camera state, from Julia in live sessions (the JS
                // OrbitControls is stale there) or from OrbitControls offline.
                const cur = () => {
                    if (live()) {
                        const s = camState.value;
                        return {
                            eye: s.slice(0, 3), tgt: s.slice(3, 6), up: s.slice(6, 9),
                            fov: s[9], ortho: s[10] > 0.5,
                        };
                    }
                    const p = c.object.position, t = c.target, u = c.object.up;
                    return {
                        eye: [p.x, p.y, p.z], tgt: [t.x, t.y, t.z], up: [u.x, u.y, u.z],
                        fov: c.object.fov, ortho: c.object.fov < perspFov - 1e-3,
                    };
                };
                // Visible half-height at the lookat plane; pan steps scale
                // with it so one click moves ~10% of the view in either mode.
                const halfH = () => {
                    const s = cur();
                    const d = len(sub(s.eye, s.tgt));
                    return (live() && s.ortho) ? d : d * Math.tan(s.fov * deg / 2);
                };
                const setPose = (eye, tgt, up) => {  // up = null keeps it
                    if (live()) {
                        camCmd.notify([1, ...eye, ...tgt, ...(up || [0, 0, 0])]);
                        return;
                    }
                    if (up) c.object.up.set(up[0], up[1], up[2]);
                    c.object.position.set(eye[0], eye[1], eye[2]);
                    c.target.set(tgt[0], tgt[1], tgt[2]);
                    c.update();
                };

                // --- camera fields ---------------------------------------
                const fields = ['phi', 'theta', 'x', 'y', 'z'].map(n => q('tv-' + n));
                // Sync Julia's camera state to BOTH OrbitControls' actual
                // position/target AND the readout boxes. Without driving
                // `c.object.position` and `c.target` here, OrbitControls'
                // initial pose (whatever Three.js baked in at load) would
                // override whatever Julia set before exporting, and the
                // first visualised frame wouldn't be the canonical view
                // the user asked for.
                const driveCamera = (s, withUp) => {
                    c.object.position.set(s.eye[0], s.eye[1], s.eye[2]);
                    c.target.set(s.tgt[0], s.tgt[1], s.tgt[2]);
                    if (withUp) c.object.up.set(s.up[0], s.up[1], s.up[2]);
                    c.update();
                };
                const syncFields = () => {
                    const s = cur();
                    const d = sub(s.eye, s.tgt), r = len(d);
                    const phi = Math.atan2(d[1], d[0]) / deg;
                    const th = Math.asin(Math.max(-1, Math.min(1, d[2] / r))) / deg;
                    const vals = [phi, th, s.eye[0], s.eye[1], s.eye[2]];
                    fields.forEach((f, i) => {
                        if (document.activeElement !== f) f.value = vals[i].toFixed(1);
                    });
                    q('tv-ortho').checked = s.ortho;
                };
                let pending = false;
                const queueSync = () => {
                    if (pending) return;
                    pending = true;
                    setTimeout(() => { pending = false; syncFields(); }, 150);
                };
                c.addEventListener('change', queueSync);  // offline mouse motion
                camState.on(queueSync);                   // live mouse motion
                // Force OrbitControls' first frame to Julia's canonical
                // eye0/la0 (the Reset target). Without this, the exported
                // Three.js scene would draw whatever baked pose its first
                // load cycle gave it, which is usually a near-front-on
                // view of the bounding-box center rather than the
                // isometric point Julia positioned for. After this snap, any
                // user-driven or `Reset`-driven change keeps the camera
                // and the readout boxes consistent.
                driveCamera(camState.value, true);
                syncFields();

                // Typed values bypass the min/max attributes; clamp here.
                const clampToInput = (v, el) => Math.min(
                    parseFloat(el.max), Math.max(parseFloat(el.min), v));
                const setAngles = (phi, th) => {
                    const s = cur();
                    const r = len(sub(s.eye, s.tgt));
                    th = Math.max(-89.9, Math.min(89.9, th));
                    const ct = Math.cos(th * deg);
                    setPose([
                        s.tgt[0] + r * ct * Math.cos(phi * deg),
                        s.tgt[1] + r * ct * Math.sin(phi * deg),
                        s.tgt[2] + r * Math.sin(th * deg),
                    ], s.tgt, null);
                };
                q('tv-phi').addEventListener('change', (e) => {
                    setAngles(clampToInput(parseFloat(e.target.value), e.target),
                              parseFloat(q('tv-theta').value));
                });
                q('tv-theta').addEventListener('change', (e) => {
                    setAngles(parseFloat(q('tv-phi').value),
                              clampToInput(parseFloat(e.target.value), e.target));
                });
                ['x', 'y', 'z'].forEach((n, i) => {
                    q('tv-' + n).addEventListener('change', (e) => {
                        const s = cur();
                        const eye = s.eye.slice();
                        eye[i] = clampToInput(parseFloat(e.target.value), e.target);
                        setPose(eye, s.tgt, null);
                    });
                });

                // --- orthographic toggle ----------------------------------
                const setOrtho = (flag) => {
                    if (live()) { camCmd.notify([2, flag ? 1 : 0]); return; }
                    const fov = flag ? ORTHO_FOV : perspFov;
                    const oldTan = Math.tan(c.object.fov * deg / 2);
                    const newTan = Math.tan(fov * deg / 2);
                    c.object.fov = fov;
                    c.object.position.lerpVectors(
                        c.target, c.object.position, oldTan / newTan);
                    c.object.updateProjectionMatrix();
                    c.update();
                };
                q('tv-ortho').addEventListener('change', (e) => setOrtho(e.target.checked));

                // --- zoom-to-cursor toggle --------------------------------
                // OrbitControls.zoomToCursor=true anchors wheel/pinch zoom to
                // the pointer; buttons always target the lookat and ignore
                // this flag (setPose() reads from cur().tgt directly).
                const applyZoomCursor = () => {
                    c.zoomToCursor = q('tv-zoomcursor').checked;
                };
                applyZoomCursor();
                q('tv-zoomcursor').addEventListener('change', applyZoomCursor);

                // --- reset / recenter -------------------------------------
                q('tv-reset').addEventListener('click', () => {
                    q('tv-ortho').checked = false;
                    if (live()) {
                        camCmd.notify([2, 0]);
                        camCmd.notify([1, ...eye0, ...la0, 0, 0, 1]);
                        return;
                    }
                    // reset() restores position/target/zoom but not fov; the
                    // saved position only frames correctly at the original fov.
                    c.object.fov = perspFov;
                    c.object.updateProjectionMatrix();
                    c.reset();
                });
                q('tv-recenter').addEventListener('click', () => {
                    setPose(cur().eye, la0, null);
                });

                // --- view presets (keep current distance) -----------------
                const isq = 1 / Math.sqrt(3);
                const presets = {
                    Iso: [[isq, isq, isq], [0, 0, 1]],
                    Front: [[0, -1, 0], [0, 0, 1]],
                    Back: [[0, 1, 0], [0, 0, 1]],
                    Left: [[-1, 0, 0], [0, 0, 1]],
                    Right: [[1, 0, 0], [0, 0, 1]],
                    Top: [[0, 0, 1], [0, 1, 0]],
                    Bottom: [[0, 0, -1], [0, 1, 0]],
                };
                for (const [name, [dir, up]] of Object.entries(presets)) {
                    q('tv-preset-' + name).addEventListener('click', () => {
                        const s = cur();
                        const d = len(sub(s.eye, s.tgt));
                        setPose(add(s.tgt, scl(dir, d)), s.tgt, up);
                    });
                }

                // --- zoom / pan cross -------------------------------------
                const zoom = (f) => {
                    const s = cur();
                    setPose(add(s.tgt, scl(sub(s.eye, s.tgt), f)), s.tgt, null);
                };
                q('tv-zoomin').addEventListener('click', () => zoom(0.9));
                q('tv-zoomout').addEventListener('click', () => zoom(1.125));
                const pan = (dx, dy) => {
                    const s = cur();
                    const view = sub(s.tgt, s.eye);
                    const right = normv(cross(view, s.up));
                    const up2 = normv(cross(right, view));
                    const st = halfH() * 0.1;
                    const off = add(scl(right, dx * st), scl(up2, dy * st));
                    setPose(add(s.eye, off), add(s.tgt, off), null);
                };
                q('tv-panleft').addEventListener('click', () => pan(-1, 0));
                q('tv-panright').addEventListener('click', () => pan(1, 0));
                q('tv-panup').addEventListener('click', () => pan(0, 1));
                q('tv-pandown').addEventListener('click', () => pan(0, -1));

                // --- save -------------------------------------------------
                // The WebGL canvas is created with preserveDrawingBuffer, so
                // toDataURL captures the last rendered frame (figure +
                // legend; the HTML controls are outside the canvas).
                q('tv-save').addEventListener('click', () => {
                    const canvas = scene.screen.canvas ||
                        scene.screen.renderer.domElement;
                    const a = document.createElement('a');
                    a.download = q('tv-savename').value || 'topopt_view.png';
                    a.href = canvas.toDataURL('image/png');
                    a.click();
                });
            }, 200);
        """
        # The observables must be interpolated as session objects (not
        # literals) so JS `.notify`/`.on`/`.value` connect to Julia; build
        # the JSCode source vector by hand since js"..." is unavailable here.
        # For static export we want a self-contained script that doesn't rely on
        # `Bonito.init_session` registering `cam_cmd`/`cam_state` (those
        # `__lookup_interpolated` keys fail without the live session). All
        # controls operate directly on `c` (Three.js OrbitControls) in
        # offline mode; the live-session push is best-effort.
        jsvec_inline(v) = "[" * join(Float64.(v), ",") * "]"
        eye0_js = jsvec_inline(eye0)
        la0_js = jsvec_inline(lookat)
        fov_js = string(persp_fov)
        scene_id_js = string(scene_id)

        container = D.div(
            container,
            D.script(
                """
                function initialize_static_view(container) {
                    const eye0 = $(eye0_js);
                    const la0 = $(la0_js);
                    const fov0 = $(fov_js);
                    const scene_id = $(string("\"", scene_id_js, "\""));

                    const ORTHO_FOV = 1.0;
                    const deg = Math.PI / 180;
                    function live() {
                        return typeof Bonito !== 'undefined' &&
                               Bonito.can_send_to_julia &&
                               Bonito.can_send_to_julia();
                    }

                    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
                    const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
                    const len = (a) => Math.hypot(a[0], a[1], a[2]);
                    const cross = (a, b) => [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    const normv = (a) => scl(a, 1 / len(a));

                    const poll = setInterval(() => {
                        if (!window.WGL || typeof window.WGL.find_scene !== 'function') return;
                        const scene = window.WGL.find_scene(scene_id);
                        if (!scene || !scene.orbitcontrols) return;
                        clearInterval(poll);
                        const c = scene.orbitcontrols;

                        // This app owns its camera in the browser. WGLMakie's
                        // live-session guard otherwise disables OrbitControls
                        // and can restore Julia's initial pose after rotation.
                        if (typeof Bonito !== 'undefined' && Bonito.can_send_to_julia) {
                            Bonito.can_send_to_julia = () => false;
                        }

                        const refreshCamera = (notify = true) => {
                            c.object.lookAt(c.target);
                            c.object.updateMatrixWorld();
                            c.update();
                            // WGLMakie disables OrbitControls updates while a
                            // Julia session is connected. Dispatching the
                            // change event still refreshes its camera matrices.
                            if (live() && notify) c.dispatchEvent({type: 'change'});
                        };

                        // Initial-frame driver: take the canonical eye0/lookat
                        // and force OrbitControls onto them so the first paint
                        // matches the Reset target. Without this Three.js keeps
                        // whatever default its first-load cycle produced.
                        const driveCamera = (eye, tgt, up) => {
                            c.object.position.set(eye[0], eye[1], eye[2]);
                            c.target.set(tgt[0], tgt[1], tgt[2]);
                            if (up) c.object.up.set(up[0], up[1], up[2]);
                            refreshCamera();
                        };

                        // setPose: drives the camera directly via OrbitControls.
                        // In offline (static) mode this is the only path and it
                        // works without any Julia session. In live mode the
                        // same OrbitControls write happens — `camera_controls`
                        // is the visual authority either way.
                        const setPose = (eye, tgt, up) => {
                            if (up) c.object.up.set(up[0], up[1], up[2]);
                            c.object.position.set(eye[0], eye[1], eye[2]);
                            c.target.set(tgt[0], tgt[1], tgt[2]);
                            refreshCamera();
                        };

                        const cur = () => {
                            const p = c.object.position, t = c.target, u = c.object.up;
                            return {
                                eye: [p.x, p.y, p.z], tgt: [t.x, t.y, t.z], up: [u.x, u.y, u.z],
                                fov: c.object.fov,
                                ortho: c.object.fov < fov0 - 1e-3,
                            };
                        };

                        const halfH = () => {
                            const s = cur();
                            const d = len(sub(s.eye, s.tgt));
                            return (live() && s.ortho) ? d : d * Math.tan(s.fov * deg / 2);
                        };

                        const fields = ['phi', 'theta', 'x', 'y', 'z'].map(n => container.querySelector('.tv-' + n));
                        const syncFields = () => {
                            const s = cur();
                            const d = sub(s.eye, s.tgt), r = len(d);
                            const phi = Math.atan2(d[1], d[0]) / deg;
                            const th = Math.asin(Math.max(-1, Math.min(1, d[2] / r))) / deg;
                            const vals = [phi, th, s.eye[0], s.eye[1], s.eye[2]];
                            fields.forEach((f, i) => {
                                if (document.activeElement !== f) f.value = vals[i].toFixed(1);
                            });
                            const ortho_el = container.querySelector('.tv-ortho');
                            if (ortho_el) ortho_el.checked = s.ortho;
                        };
                        let pending = false;
                        const queueSync = () => {
                            if (pending) return;
                            pending = true;
                            setTimeout(() => { pending = false; syncFields(); }, 150);
                        };
                        c.addEventListener('change', queueSync);
                        syncFields();

                        const clampToInput = (v, el) =>
                            Math.min(parseFloat(el.max), Math.max(parseFloat(el.min), v));
                        const setAngles = (phi, th) => {
                            const s = cur();
                            const r = len(sub(s.eye, s.tgt));
                            th = Math.max(-89.9, Math.min(89.9, th));
                            const ct = Math.cos(th * deg);
                            setPose([
                                s.tgt[0] + r * ct * Math.cos(phi * deg),
                                s.tgt[1] + r * ct * Math.sin(phi * deg),
                                s.tgt[2] + r * Math.sin(th * deg),
                            ], s.tgt, null);
                        };
                        const fldOrNull = (sel) => {
                            const el = container.querySelector(sel);
                            return el ? el : null;
                        };
                        const phi_el = fldOrNull('.tv-phi');
                        if (phi_el) phi_el.addEventListener('change', (e) => {
                            const theta_el = fldOrNull('.tv-theta');
                            const tv = (theta_el && parseFloat(theta_el.value)) || 30;
                            setAngles(clampToInput(parseFloat(e.target.value), e.target), tv);
                        });
                        const theta_el = fldOrNull('.tv-theta');
                        if (theta_el) theta_el.addEventListener('change', (e) => {
                            const phi_el2 = fldOrNull('.tv-phi');
                            const pv = (phi_el2 && parseFloat(phi_el2.value)) || 45;
                            setAngles(pv, clampToInput(parseFloat(e.target.value), e.target));
                        });
                        ['x', 'y', 'z'].forEach((n, i) => {
                            const el = fldOrNull('.tv-' + n);
                            if (!el) return;
                            const applyCoordinate = () => {
                                const value = parseFloat(el.value);
                                if (!Number.isFinite(value)) return;
                                const s = cur();
                                const eye = s.eye.slice();
                                eye[i] = clampToInput(value, el);
                                el.value = eye[i].toFixed(1);
                                setPose(eye, s.tgt, null);
                            };
                            el.addEventListener('change', applyCoordinate);
                            el.addEventListener('keydown', (e) => {
                                if (e.key === 'Enter') {
                                    e.preventDefault();
                                    applyCoordinate();
                                    el.blur();
                                }
                            });
                        });

                        const setOrtho = (flag) => {
                            const fov = flag ? ORTHO_FOV : fov0;
                            const oldTan = Math.tan(c.object.fov * deg / 2);
                            const newTan = Math.tan(fov * deg / 2);
                            c.object.fov = fov;
                            c.object.position.lerpVectors(c.target, c.object.position, oldTan / newTan);
                            c.object.updateProjectionMatrix();
                            refreshCamera();
                        };
                        const ortho_el = fldOrNull('.tv-ortho');
                        if (ortho_el) ortho_el.addEventListener('change', (e) => setOrtho(e.target.checked));

                        const zoomcursor_el = fldOrNull('.tv-zoomcursor');
                        if (zoomcursor_el) {
                            const applyZoomCursor = () => {
                                c.zoomToCursor = zoomcursor_el.checked;
                            };
                            applyZoomCursor();
                            zoomcursor_el.addEventListener('change', applyZoomCursor);
                        }

                        const reset_el = fldOrNull('.tv-reset');
                        if (reset_el) reset_el.addEventListener('click', () => {
                            const ortho_el2 = fldOrNull('.tv-ortho');
                            if (ortho_el2) ortho_el2.checked = false;
                            c.object.fov = fov0;
                            c.object.updateProjectionMatrix();
                            setPose(eye0, la0, [0, 0, 1]);
                        });
                        const recenter_el = fldOrNull('.tv-recenter');
                        if (recenter_el) recenter_el.addEventListener('click', () => {
                            const s = cur();
                            setPose(s.eye, la0, null);
                        });

                        const isq = 1 / Math.sqrt(3);
                        const presets = {
                            Iso: [[isq, isq, isq], [0, 0, 1]],
                            Front: [[0, -1, 0], [0, 0, 1]],
                            Back: [[0, 1, 0], [0, 0, 1]],
                            Left: [[-1, 0, 0], [0, 0, 1]],
                            Right: [[1, 0, 0], [0, 0, 1]],
                            Top: [[0, 0, 1], [0, 1, 0]],
                            Bottom: [[0, 0, -1], [0, 1, 0]],
                        };
                        Object.entries(presets).forEach(([name, [dir, up]]) => {
                            const btn = fldOrNull('.tv-preset-' + name);
                            if (!btn) return;
                            btn.addEventListener('click', () => {
                                const s = cur();
                                const d = len(sub(s.eye, s.tgt));
                                setPose(add(s.tgt, scl(dir, d)), s.tgt, up);
                            });
                        });

                        const zoom = (f) => {
                            const s = cur();
                            setPose(add(s.tgt, scl(sub(s.eye, s.tgt), f)), s.tgt, null);
                        };
                        const zoomin_el = fldOrNull('.tv-zoomin');
                        if (zoomin_el) zoomin_el.addEventListener('click', () => zoom(0.9));
                        const zoomout_el = fldOrNull('.tv-zoomout');
                        if (zoomout_el) zoomout_el.addEventListener('click', () => zoom(1.125));

                        const pan = (dx, dy) => {
                            const s = cur();
                            const view = sub(s.tgt, s.eye);
                            const right = normv(cross(view, s.up));
                            const up2 = normv(cross(right, view));
                            const st = halfH() * 0.1;
                            const off = add(scl(right, dx * st), scl(up2, dy * st));
                            setPose(add(s.eye, off), add(s.tgt, off), null);
                        };
                        const pan_btns = [
                            ['panleft', -1, 0],
                            ['panright', 1, 0],
                            ['panup', 0, 1],
                            ['pandown', 0, -1],
                        ];
                        pan_btns.forEach(([cls, dx, dy]) => {
                            const btn = fldOrNull('.tv-' + cls);
                            if (!btn) return;
                            btn.addEventListener('click', () => pan(dx, dy));
                        });

                        const save_el = fldOrNull('.tv-save');
                        if (save_el) save_el.addEventListener('click', () => {
                            const canvas = scene.screen.canvas || scene.screen.renderer.domElement;
                            const save_box = fldOrNull('.tv-savename');
                            const a = document.createElement('a');
                            a.download = (save_box && save_box.value) || 'topopt_view.png';
                            a.href = canvas.toDataURL('image/png');
                            a.click();
                        });

                        // Final: drive OrbitControls into Julia's eye0/lookat
                        // so the very first paint matches the Reset target.
                        driveCamera(eye0, la0, [0, 0, 1]);
                    }, 200);
                }
                // Module scripts do not expose document.currentScript. Find
                // this script by its unique function marker and use its
                // parent as the control/figure container.
                const self = [...document.scripts].find(s =>
                    s.textContent.includes('initialize_static_view'));
                initialize_static_view(self ? self.parentElement : document.body);
                """;
                type="module",
            ),
        )

        return container
    end
end
end
