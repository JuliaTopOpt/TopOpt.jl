using Ferrite: Cell

"""
    abstract type StiffnessTopOptProblem{dim, T} <: AbstractTopOptProblem end

An abstract stiffness topology optimization problem. All subtypes must have the following fields:
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
abstract type StiffnessTopOptProblem{dim,T} <: AbstractTopOptProblem end

# Fallbacks
"""
    getdim(problem::StiffnessTopOptProblem)

Return the spatial dimension of the problem.
"""
getdim(::StiffnessTopOptProblem{dim,T}) where {dim,T} = dim

"""
    floattype(problem::StiffnessTopOptProblem)

Return the number type used for the problem's coordinates and computations.
"""
floattype(::StiffnessTopOptProblem{dim,T}) where {dim,T} = T
getE(p::StiffnessTopOptProblem) = p.E
getν(p::StiffnessTopOptProblem) = p.ν

"""
    getplanestress(p::StiffnessTopOptProblem)

Return `true` if the problem uses the plane-stress constitutive law (λ =
Eν/(1-ν²)) and `false` for plane strain (λ = Eν/((1+ν)(1-2ν))). Both share
the same shear modulus 2μ = E/(1+ν); only λ differs. Defaults to `false`
(plane strain), matching the historical TopOpt convention.
"""
getplanestress(::StiffnessTopOptProblem) = false

"""
    getkσpoint(p::StiffnessTopOptProblem)

Where the stress used to build the geometric stiffness Kσ is evaluated:
`:quadrature` (default — the stress is evaluated at every quadrature
point) or `:centroid` (a single evaluation at the element centroid, the
superconvergent stress point used by the topBuck250 benchmark of Ferrari,
Sigmund and Guest 2021).
"""
getkσpoint(::StiffnessTopOptProblem) = :quadrature

"""
    YoungsModulus(p::StiffnessTopOptProblem)

Return the Young's modulus of a structural topology optimization problem.
"""
YoungsModulus(p::StiffnessTopOptProblem) = getE(p)

"""
    PoissonRatio(p::StiffnessTopOptProblem)

Return the Poisson's ratio of a structural topology optimization problem.
"""
PoissonRatio(p::StiffnessTopOptProblem) = getν(p)
getgeomorder(p::StiffnessTopOptProblem) = nnodespercell(p) in (9, 27) ? 2 : 1
getdensity(::StiffnessTopOptProblem{dim,T}) where {dim,T} = T(0)
getmetadata(p::StiffnessTopOptProblem) = p.metadata

"""
    getdh(problem::StiffnessTopOptProblem)

Return the `Ferrite.DofHandler` of the problem.
"""
getdh(p::StiffnessTopOptProblem) = p.ch.dh

"""
    getcloaddict(problem::StiffnessTopOptProblem)

Return the concentrated-load dictionary (node index => load vector).
"""
getcloaddict(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,Vector{T}}()

"""
    getpressuredict(problem::StiffnessTopOptProblem)

Return the pressure-load dictionary (faceset name => pressure value).
"""
getpressuredict(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,T}()

"""
    getfacesets(problem::StiffnessTopOptProblem)

Return the face sets of the problem's grid.
"""
getfacesets(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,Tuple{Int,T}}()
Ferrite.getncells(problem::StiffnessTopOptProblem) = Ferrite.getncells(getdh(problem).grid)

"""
    PointLoadCantilever{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}

A cantilever beam problem with a point load applied at the free end. Available
in 2D and 3D.

```
///**********************************
///*                                *
///*                                * |
///*                                * |
///********************************** v


struct PointLoadCantilever{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
    rect_grid::RectilinearGrid{dim, T, N, M}
    E::T
    ν::T
    ch::ConstraintHandler{<:DofHandler, T}
    force::T
    force_dof::Integer
    metadata::Metadata
end
```

- `dim`: dimension of the problem
- `T`: number type for computations and coordinates
- `N`: number of nodes in a cell of the grid
- `M`: number of faces in a cell of the grid
- `rect_grid`: a RectilinearGrid struct
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `force_dof`: dof number at which the force is applied
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
struct PointLoadCantilever{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler,T},
    Tf<:Integer,
    Tm<:Metadata,
} <: StiffnessTopOptProblem{dim,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::PointLoadCantilever)
    return println(io, "TopOpt point load cantilever beam problem")
end
getplanestress(p::PointLoadCantilever) = p.planestress
getkσpoint(p::PointLoadCantilever) = p.kσpoint

"""
    PointLoadCantilever(nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force; celltype=:Linear) where {dim}

- `dim`: dimension of the problem
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `nels`: number of elements in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `sizes`: the size of each element in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `celltype`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

Example:
```
nels = (60,20);
sizes = (1.0,1.0);
E = 1.0;
ν = 0.3;
force = 1.0;

# Linear elements and linear basis functions
celltype = :Linear

# Quadratic elements and quadratic basis functions
#celltype = :Quadratic

problem = PointLoadCantilever(nels, sizes, E, ν, force; celltype)
```
"""
function PointLoadCantilever(
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0;
    celltype::Symbol=:Linear,
    planestress::Bool=false,
    kσpoint::Symbol=:quadrature,
) where {dim}
    return _PointLoadCantilever(
        _celltype_tag(celltype), nels, sizes, E, ν, force, planestress, kσpoint
    )
end
function _PointLoadCantilever(
    ::Val{CellType},
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0,
    planestress=false,
    kσpoint=:quadrature,
) where {dim,CellType}
    iseven(nels[2]) && (length(nels) < 3 || iseven(nels[3])) ||
        throw("Grid does not have an even number of elements along the y and/or z axes.")

    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force)))
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Linear)
    else
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Quadratic)
    end

    if haskey(rect_grid.grid.facetsets, "fixed_all")
        pop!(rect_grid.grid.facetsets, "fixed_all")
    end
    #addfaceset!(rect_grid.grid, "fixed_all", x -> left(rect_grid, x));
    addnodeset!(rect_grid.grid, "fixed_all", x -> left(rect_grid, x))

    if haskey(rect_grid.grid.nodesets, "down_force")
        pop!(rect_grid.grid.nodesets, "down_force")
    end
    addnodeset!(
        rect_grid.grid, "down_force", x -> right(rect_grid, x) && middley(rect_grid, x)
    )

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    refshape = Ferrite.getrefshape(eltype(rect_grid.grid.cells))
    if CellType === :Linear || dim === 3
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^dim) # Add a displacement field
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^dim) # Add a displacement field        
    end
    close!(dh)

    ch = ConstraintHandler(dh)

    #dbc = Dirichlet(:u, getfacetset(rect_grid.grid, "fixed_all"), (x,t) -> zeros(T, dim), collect(1:dim))
    dbc = Dirichlet(
        :u, getnodeset(rect_grid.grid, "fixed_all"), (x, t) -> zeros(T, dim), collect(1:dim)
    )
    add!(ch, dbc)
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    fnode = Tuple(getnodeset(rect_grid.grid, "down_force"))[1]
    node_dofs = metadata.node_dofs
    force_dof = node_dofs[2, fnode]

    N = nnodespercell(rect_grid)
    M = nfacespercell(rect_grid)

    return PointLoadCantilever(
        rect_grid, E, ν, ch, force, force_dof, metadata, planestress, kσpoint
    )
end

"""
    CompressedColumn(nels::NTuple{2,Int}, sizes::NTuple{2}, E=1.0, ν=0.3, q=1.0e-3, bfrac=1/15; celltype=:Linear)

The 2D compressed column benchmark of [FerrariSigmundGuest2021](@cite)
(Example 5.1, the `topBuck250` code): a cantilever of `nels` elements
clamped on the left edge, subject to a compressive (negative x) load of
total magnitude `q` spread over the segment `b = bfrac * Ly` centered at
mid-height of the right edge, with a trapezoidal nodal distribution
(halved at the segment ends) matching the reference implementation.

Arguments:

- `nels`: number of elements in each direction, a 2-tuple
- `sizes`: the size of each element in each direction, a 2-tuple
- `E`: Young's modulus
- `ν`: Poisson's ratio
- `q`: total magnitude of the compressive load
- `bfrac`: load segment length as a fraction of the height `Ly`
  (`bfrac >= 1` loads the whole edge uniformly)
- `celltype`: can be `:Linear` (default)
- `planestress`: use the plane-stress constitutive law (default `false`,
  plane strain)
- `kσpoint`: `:quadrature` (default) or `:centroid` (superconvergent
  centroid stress, the topBuck250 convention) for the geometric stiffness
- `passive_solid`: `(cols, rows)` of fixed solid elements at the loaded edge
  around the load (the topBuck250 passive block)
- `vertical`: clamp the bottom edge and compress from the top edge
  (default `false`: clamped left, compressed from the right edge)

See also [`buckling`](@ref) and [`BucklingConeFun`](@ref) for the buckling
analysis used with this problem.

```
   y ▲                        b = bfrac·Ly
     │                 ┌───────────────┐
     │                 │               │
     │                 │               │
     │                 │               │  ← ← ←  compressive load q
     │                 │               │         (segment at mid-height
     │                 │               │          of the right edge)
     │                 │               │
     │                 │               │
     │                 │               │
     │                 │               │
     └─────────────────┴───────────────┴──► x
     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
     ▒▒ clamped left edge (u_x = u_y = 0)
     ▒▒
     ◄────────────── Lx ──────────────►
```
"""
struct CompressedColumn{T,N,M,Tr<:RectilinearGrid{2,T,N,M},Tc,Td,Tm} <:
       StiffnessTopOptProblem{2,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    cloads::Td
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
    passive::Vector{Int}
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::CompressedColumn)
    return println(io, "TopOpt compressed column problem")
end
getplanestress(p::CompressedColumn) = p.planestress
getkσpoint(p::CompressedColumn) = p.kσpoint

"""
    getpassive(p::CompressedColumn)

Indices of the passive (fixed solid) elements near the load, `x = 1` there.
"""
getpassive(p::CompressedColumn) = p.passive
function CompressedColumn(
    nels::NTuple{2,Int},
    sizes::NTuple{2},
    E=1.0,
    ν=0.3,
    q=1.0e-3,
    bfrac=1 / 15;
    celltype::Symbol=:Linear,
    planestress::Bool=false,
    kσpoint::Symbol=:quadrature,
    passive_solid::NTuple{2,Int}=(0, 0),
    vertical::Bool=false,
)
    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(q), typeof(bfrac)))
    rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Linear)
    Ly = rect_grid.corners[2][2] - rect_grid.corners[1][2]
    Lx = rect_grid.corners[2][1] - rect_grid.corners[1][1]
    midy = rect_grid.corners[1][2] + Ly / 2
    midx = rect_grid.corners[1][1] + Lx / 2

    # Clamped edge and load segment: horizontal (clamped left, compressed
    # from the right edge) or vertical (clamped base, compressed from the
    # top edge).
    if vertical
        if haskey(rect_grid.grid.nodesets, "fixed_base")
            pop!(rect_grid.grid.nodesets, "fixed_base")
        end
        addnodeset!(rect_grid.grid, "fixed_base", x -> bottom(rect_grid, x))
        if haskey(rect_grid.grid.nodesets, "comp_load")
            pop!(rect_grid.grid.nodesets, "comp_load")
        end
        addnodeset!(
            rect_grid.grid,
            "comp_load",
            x -> top(rect_grid, x) && abs(x[1] - midx) <= bfrac * Lx / 2 + 1e-9,
        )
    else
        if haskey(rect_grid.grid.nodesets, "fixed_left")
            pop!(rect_grid.grid.nodesets, "fixed_left")
        end
        addnodeset!(rect_grid.grid, "fixed_left", x -> left(rect_grid, x))
        if haskey(rect_grid.grid.nodesets, "comp_load")
            pop!(rect_grid.grid.nodesets, "comp_load")
        end
        addnodeset!(
            rect_grid.grid,
            "comp_load",
            x -> right(rect_grid, x) && abs(x[2] - midy) <= bfrac * Ly / 2 + 1e-9,
        )
    end

    # displacement field + clamped edge (mirrors _PointLoadCantilever)
    dh = DofHandler(rect_grid.grid)
    refshape = Ferrite.getrefshape(eltype(rect_grid.grid.cells))
    if celltype === :Linear
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^2)
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^2)
    end
    close!(dh)
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(
        :u,
        getnodeset(rect_grid.grid, vertical ? "fixed_base" : "fixed_left"),
        (x, t) -> zeros(T, 2),
        collect(1:2),
    )
    add!(ch, dbc)
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    # compressive load: total q, trapezoidal (halved at the segment ends).
    # A segment covering the whole edge (bfrac >= 1) is distributed
    # uniformly, matching full-edge load setups.
    loadnodes = sort(
        collect(getnodeset(rect_grid.grid, "comp_load"));
        by=ni -> get_node_coordinate(rect_grid.grid.nodes[ni])[vertical ? 1 : 2],
    )
    nseg = length(loadnodes)
    fulledge = bfrac >= 1.0
    w = fulledge ? T(q / nseg) : T(q / max(nseg - 1, 1))
    cloads = Dict{Int,Vector{T}}()
    for (k, ni) in enumerate(loadnodes)
        wi = fulledge ? w : ((k == 1 || k == nseg) ? w / 2 : w)
        cloads[ni] = vertical ? [zero(T), -wi] : [-wi, zero(T)]
    end
    # passive solid block near the load (topBuck250's 10×20 elements at
    # 480×240): `passive_solid` columns × rows at the loaded edge, centered
    # along the edge.
    passive = Int[]
    if vertical
        pcol = (nels[1] ÷ 2 .+ (-(passive_solid[1] ÷ 2 - 1))):(passive_solid[1] ÷ 2)
        prow = nels[2] .- (0:(passive_solid[2] - 1))
        for ix in pcol, iy in prow
            (1 <= ix <= nels[1] && 1 <= iy <= nels[2]) || continue
            push!(passive, (ix - 1) * nels[2] + iy)
        end
    else
        pcol = nels[1] .- (0:(passive_solid[1] - 1))
        prow = (nels[2] ÷ 2 .+ (-(passive_solid[2] ÷ 2 - 1))):(passive_solid[2] ÷ 2)
        for ix in pcol, iy in prow
            (1 <= ix <= nels[1] && 1 <= iy <= nels[2]) || continue
            push!(passive, (ix - 1) * nels[2] + iy)
        end
    end
    unique!(sort!(passive))
    return CompressedColumn(
        rect_grid, E, ν, ch, cloads, metadata, planestress, kσpoint, passive
    )
end
nnodespercell(p::CompressedColumn) = nnodespercell(p.rect_grid)
getcloaddict(p::CompressedColumn) = p.cloads

"""
    HalfMBB{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}

The half Messerschmitt-Bölkow-Blohm (MBB) beam problem with a point load at
the top-left corner. Available in 2D and 3D.

```
 |
 |
 v
O*********************************
O*                               *
O*                               *
O*                               *
O*********************************
                                 O


struct HalfMBB{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
    rect_grid::RectilinearGrid{dim, T, N, M}
    E::T
    ν::T
    ch::ConstraintHandler{<:DofHandler, T}
    force::T
    force_dof::Integer
    metadata::Metadata
end
```

- `dim`: dimension of the problem
- `T`: number type for computations and coordinates
- `N`: number of nodes in a cell of the grid
- `M`: number of faces in a cell of the grid
- `rect_grid`: a RectilinearGrid struct
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the top left of half the MBB (positive is downward)
- `force_dof`: dof number at which the force is applied
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
struct HalfMBB{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler,T},
    Tf<:Integer,
    Tm<:Metadata,
} <: StiffnessTopOptProblem{dim,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::HalfMBB)
    return println(io, "TopOpt half MBB problem")
end
getplanestress(p::HalfMBB) = p.planestress
getkσpoint(p::HalfMBB) = p.kσpoint

"""
    HalfMBB(nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force; celltype=:Linear) where {dim}

- `dim`: dimension of the problem
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the top left of half the MBB (positive is downward)
- `nels`: number of elements in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `sizes`: the size of each element in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `celltype`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

Example:
```
nels = (60,20);
sizes = (1.0,1.0);
E = 1.0;
ν = 0.3;
force = -1.0;

# Linear elements and linear basis functions
celltype = :Linear

# Quadratic elements and quadratic basis functions
#celltype = :Quadratic

problem = HalfMBB(nels, sizes, E, ν, force; celltype)
```
"""
function HalfMBB(
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0;
    celltype::Symbol=:Linear,
    planestress::Bool=false,
    kσpoint::Symbol=:quadrature,
) where {dim}
    return _HalfMBB(_celltype_tag(celltype), nels, sizes, E, ν, force, planestress, kσpoint)
end
function _HalfMBB(
    ::Val{CellType},
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0,
    planestress=false,
    kσpoint=:quadrature,
) where {dim,CellType}
    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force)))
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Linear)
    else
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Quadratic)
    end

    if haskey(rect_grid.grid.facetsets, "fixed_u1")
        pop!(rect_grid.grid.facetsets, "fixed_u1")
    end
    #addfaceset!(rect_grid.grid, "fixed_u1", x -> left(rect_grid, x));
    addnodeset!(rect_grid.grid, "fixed_u1", x -> left(rect_grid, x))

    if haskey(rect_grid.grid.nodesets, "fixed_u2")
        pop!(rect_grid.grid.nodesets, "fixed_u2")
    end
    addnodeset!(
        rect_grid.grid, "fixed_u2", x -> bottom(rect_grid, x) && right(rect_grid, x)
    )

    if haskey(rect_grid.grid.nodesets, "down_force")
        pop!(rect_grid.grid.nodesets, "down_force")
    end
    addnodeset!(rect_grid.grid, "down_force", x -> top(rect_grid, x) && left(rect_grid, x))

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    refshape = Ferrite.getrefshape(eltype(rect_grid.grid.cells))
    if CellType === :Linear || dim === 3
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^dim)
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^dim)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    #dbc1 = Dirichlet(:u, getfacetset(rect_grid.grid, "fixed_u1"), (x,t)->T[0], [1])
    dbc1 = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed_u1"), (x, t) -> T[0], [1])
    add!(ch, dbc1)
    dbc2 = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed_u2"), (x, t) -> T[0], [2])
    add!(ch, dbc2)
    close!(ch)

    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    fnode = Tuple(getnodeset(rect_grid.grid, "down_force"))[1]
    node_dofs = metadata.node_dofs
    force_dof = node_dofs[2, fnode]

    N = nnodespercell(rect_grid)
    M = nfacespercell(rect_grid)

    return HalfMBB(rect_grid, E, ν, ch, force, force_dof, metadata, planestress, kσpoint)
end

nnodespercell(p::Union{PointLoadCantilever,HalfMBB}) = nnodespercell(p.rect_grid)
function getcloaddict(p::Union{PointLoadCantilever{dim,T},HalfMBB{dim,T}}) where {dim,T}
    f = T[0, -p.force, 0]
    fnode = Tuple(getnodeset(p.rect_grid.grid, "down_force"))[1]
    return Dict{Int,Vector{T}}(fnode => f)
end

"""
    LBeam{T, N, M} <: StiffnessTopOptProblem{2, T}

An L-shaped beam problem with a point load at the free end of the lower slab.
Always 2D.

```
////////////
............
.          .
.          .
.          . 
.          .                    
.          ......................
.                               .
.                               . 
.                               . |
................................. v
                                force

struct LBeam{T, N, M} <: StiffnessTopOptProblem{2, T}
    E::T
    ν::T
    ch::ConstraintHandler{<:DofHandler, T}
    force::T
    force_dof::Integer
    metadata::Metadata
end
```

- `T`: number type for computations and coordinates
- `N`: number of nodes in a cell of the grid
- `M`: number of faces in a cell of the grid
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `force_dof`: dof number at which the force is applied
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
struct LBeam{T,N,M,Tc<:ConstraintHandler{<:DofHandler,T},Tf<:Integer,Tm<:Metadata} <:
       StiffnessTopOptProblem{2,T}
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::LBeam)
    return println(io, "TopOpt L-beam problem")
end
getplanestress(p::LBeam) = p.planestress
getkσpoint(p::LBeam) = p.kσpoint

"""
    LBeam(::Type{T}=Float64; celltype=:Linear, length = 100, height = 100, upperslab = 50, lowerslab = 50, E = 1.0, ν = 0.3, force = 1.0, load_width = nothing) where {T}

- `T`: number type for computations and coordinates
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `load_width`: if an integer `w ≥ 1`, the force is distributed equally over
  `w` consecutive nodes of the right edge centered on the midpoint, instead of
  the default single midpoint node. Distributing the load removes the
  point-load stress singularity, which otherwise dominates stress-based
  optimization (a mesh-dependent artifact of load introduction, not of the
  topology).
- `length`, `height`, `upperslab` and `lowerslab` are explained in [`LGrid`](@ref).
- `celltype`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

Example:
```
E = 1.0;
ν = 0.3;
force = 1.0;

# Linear elements and linear basis functions
celltype = :Linear

# Quadratic elements and quadratic basis functions
#celltype = :Quadratic

problem = LBeam(E = E, ν = ν, force = force)
```
"""
function LBeam(
    (::Type{T})=Float64;
    celltype::Symbol=:Linear,
    length=100,
    height=100,
    upperslab=50,
    lowerslab=50,
    E=1.0,
    ν=0.3,
    force=1.0,
    load_width=nothing,
    planestress=false,
    kσpoint=:quadrature,
) where {T}
    return _LBeam(
        _celltype_tag(celltype),
        T;
        length=length,
        height=height,
        upperslab=upperslab,
        lowerslab=lowerslab,
        E=E,
        ν=ν,
        force=force,
        load_width=load_width,
        planestress=planestress,
        kσpoint=kσpoint,
    )
end
function _LBeam(
    ::Val{CellType},
    ::Type{T};
    length=100,
    height=100,
    upperslab=50,
    lowerslab=50,
    E=1.0,
    ν=0.3,
    force=1.0,
    load_width=nothing,
    planestress=false,
    kσpoint=:quadrature,
) where {CellType,T}
    # Create displacement field u
    grid = LGrid(
        T;
        celltype=CellType,
        length=length,
        height=height,
        upperslab=upperslab,
        lowerslab=lowerslab,
        load_width=load_width,
    )

    dh = DofHandler(grid)
    refshape = Ferrite.getrefshape(eltype(grid.cells))
    if CellType === :Linear
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^2)
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^2)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> T[0, 0], [1, 2])
    add!(ch, dbc)
    close!(ch)

    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    fnode = Tuple(getnodeset(grid, "load"))[1]
    node_dofs = metadata.node_dofs
    force_dof = node_dofs[2, fnode]

    N = nnodes(eltype(grid.cells))
    M = Ferrite.nfacets(Ferrite.getrefshape(eltype(grid.cells)))
    return LBeam{T,N,M,typeof(ch),typeof(force_dof),typeof(metadata)}(
        E, ν, ch, force, force_dof, metadata, planestress, kσpoint
    )
end

function boundingbox(nodes::Vector{Node{dim,T}}) where {dim,T}
    xmin1 = minimum(n -> n.x[1], nodes)
    xmax1 = maximum(n -> n.x[1], nodes)
    xmin2 = minimum(n -> n.x[2], nodes)
    xmax2 = maximum(n -> n.x[2], nodes)
    if dim === 2
        return ((xmin1, xmin2), (xmax1, xmax2))
    else
        xmin3 = minimum(n -> n.x[3], nodes)
        xmax3 = maximum(n -> n.x[3], nodes)
        return ((xmin1, xmin2, xmin3), (xmax1, xmax2, xmax3))
    end
end

function boundingbox(grid::Ferrite.Grid{dim}) where {dim}
    return boundingbox(grid.nodes)
end

function RectilinearTopology(b, topology=ones(getncells(getdh(b).grid)))
    bb = boundingbox(getdh(b).grid)
    go = getgeomorder(b)
    nels = Int.(round.(bb[2] .- bb[1]))
    dim = length(nels)
    if go === 1
        rectgrid = generate_grid(Quadrilateral, nels, Vec{dim}(bb[1]), Vec{dim}(bb[2]))
    elseif go === 2
        rectgrid = generate_grid(
            QuadraticQuadrilateral, nels, Vec{dim}(bb[1]), Vec{dim}(bb[2])
        )
    else
        throw("Unsupported geometry.")
    end
    new_topology = zeros(prod(nels))
    for (i, cell) in enumerate(CellIterator(getdh(b)))
        sub = Int.(round.((cell.coords[1]...,))) .+ (1, 1)
        ind = LinearIndices(nels)[sub...]
        new_topology[ind] = topology[i]
    end
    return copy(reshape(new_topology, nels)')
end

nnodespercell(p::LBeam{T,N}) where {T,N} = N
getdim(::LBeam) = 2
function getcloaddict(p::LBeam{T}) where {T}
    load_nodes = getnodeset(getdh(p).grid, "load")
    # Equal split of the resultant force over the load node(s); one node
    # unless the grid was built with `load_width` (distributed load).
    f = T[0, -p.force / length(load_nodes)]
    return Dict{Int,Vector{T}}(n => copy(f) for n in load_nodes)
end

"""
    TieBeam{T, N, M} <: StiffnessTopOptProblem{2, T}

A tie-beam problem with distributed loading on specified elements. Always 2D.

```
                                                               1
                                                               
                                                              OOO
                                                              ...
                                                              . .
                                                           4  . . 
                                30                            . .   
/ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <-
/ .                                                                 . <- 2 f 
/ .    3                                                            . <- 
/ . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . <-
                                                              ^^^
                                                              |||
                                                              1 f

struct TieBeam{T, N, M} <: StiffnessTopOptProblem{2, T}
    E::T
    ν::T
    force::T
    ch::ConstraintHandler{<:DofHandler, T}
    metadata::Metadata
end
```

- `T`: number type for computations and coordinates
- `N`: number of nodes in a cell of the grid
- `M`: number of faces in a cell of the grid
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
struct TieBeam{T,N,M,Tc<:ConstraintHandler{<:DofHandler,T},Tm<:Metadata} <:
       StiffnessTopOptProblem{2,T}
    E::T
    ν::T
    force::T
    ch::Tc
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TieBeam)
    return println(io, "TopOpt tie-beam problem")
end
getplanestress(p::TieBeam) = p.planestress
getkσpoint(p::TieBeam) = p.kσpoint

"""
    TieBeam(::Type{T} = Float64; celltype=:Linear, refine = 1, force = T(1), E = T(1), ν = T(0.3)) where {T}

- `T`: number type for computations and coordinates
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `refine`: an integer value of 1 or greater that specifies the mesh refinement extent. A value of 1 gives the standard tie-beam problem in literature.
- `celltype`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.
"""
function TieBeam(
    (::Type{T})=Float64;
    celltype::Symbol=:Linear,
    refine=1,
    force=T(1),
    E=T(1),
    ν=T(0.3),
    planestress=false,
    kσpoint=:quadrature,
) where {T}
    return _TieBeam(
        _celltype_tag(celltype),
        T;
        refine=refine,
        force=force,
        E=E,
        ν=ν,
        planestress=planestress,
        kσpoint=kσpoint,
    )
end
function _TieBeam(
    ::Val{CellType},
    ::Type{T};
    refine=1,
    force=T(1),
    E=T(1),
    ν=T(0.3),
    planestress=false,
    kσpoint=:quadrature,
) where {T,CellType}
    grid = TieBeamGrid(T; celltype=CellType, refine=refine)

    dh = DofHandler(grid)
    refshape = Ferrite.getrefshape(eltype(grid.cells))
    if CellType === :Linear
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^2)
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^2)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfacetset(grid, "leftfixed"), (x, t) -> T[0, 0], [1, 2])
    add!(ch, dbc)
    close!(ch)

    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    N = nnodes(eltype(grid.cells))
    M = Ferrite.nfacets(Ferrite.getrefshape(eltype(grid.cells)))
    return TieBeam{T,N,M,typeof(ch),typeof(metadata)}(
        E, ν, force, ch, metadata, planestress, kσpoint
    )
end

getdim(::TieBeam) = 2
nnodespercell(::TieBeam{T,N}) where {T,N} = N
function getpressuredict(p::TieBeam{T}) where {T}
    return Dict{String,T}("rightload" => 2 * p.force, "bottomload" => -p.force)
end
getfacesets(p::TieBeam) = getdh(p).grid.facetsets

# ============================================================================
# Heat Transfer Problem Types
# ============================================================================

"""
    abstract type HeatTransferTopOptProblem{dim, T} <: AbstractTopOptProblem end

An abstract heat transfer topology optimization problem for steady-state heat conduction.

Governing equation: -∇·(k(ρ)∇T) = q    in Ω
                   T = T_D            on Γ_D (Dirichlet BC)
                   k∇T·n = q_N        on Γ_N (Neumann BC)

SIMP interpolation: k(ρ) = k_min + ρ^p (k_0 - k_min)
Heat source q is NOT penalized (external input, not a material property).

Mathematical note: For thermal compliance J = Q^T T, the gradient is:
    dJ/dx_e = -T_e^T Ke T_e · dρ_e/dx_e
This is the same form as structural compliance because Q doesn't depend on x.

See [BendsoeSigmund2003](@cite) §1.3 and §4.1 for thermal topology
optimization, and [Iga2009](@cite) for SIMP-based heat conduction.

All subtypes must have:
- `ch`: ConstraintHandler with temperature DOFs (1 DOF per node)
- `metadata`: Metadata with cell-node-dof relationships
- `k`: thermal conductivity
- `heatfluxdict`: surface heat flux on boundaries (Dict{String,Float64})
"""
abstract type HeatTransferTopOptProblem{dim,T} <: AbstractTopOptProblem end

# Fallbacks for HeatTransferTopOptProblem
getdim(::HeatTransferTopOptProblem{dim,T}) where {dim,T} = dim
floattype(::HeatTransferTopOptProblem{dim,T}) where {dim,T} = T

"""
    getk(problem::HeatTransferTopOptProblem)

Return the thermal conductivity of a heat transfer problem.
"""
getk(p::HeatTransferTopOptProblem) = p.k
getmetadata(p::HeatTransferTopOptProblem) = p.metadata
getdh(p::HeatTransferTopOptProblem) = p.ch.dh
getpressuredict(p::HeatTransferTopOptProblem{dim,T}) where {dim,T} = Dict{String,T}()

"""
    getheatfluxdict(problem::HeatTransferTopOptProblem)

Return the surface heat-flux dictionary (faceset name => heat flux value).
"""
getheatfluxdict(p::HeatTransferTopOptProblem{dim,T}) where {dim,T} = Dict{String,T}()
getfacesets(p::HeatTransferTopOptProblem) = getdh(p).grid.facetsets
function Ferrite.getncells(problem::HeatTransferTopOptProblem)
    return Ferrite.getncells(getdh(problem).grid)
end
getgeomorder(p::HeatTransferTopOptProblem) = nnodespercell(p) in (9, 27) ? 2 : 1
getdensity(::HeatTransferTopOptProblem{dim,T}) where {dim,T} = T(0)
getcloaddict(p::HeatTransferTopOptProblem{dim,T}) where {dim,T} = Dict{Int,Vector{T}}()

"""
    struct HeatConductionProblem{dim, T, N, M} <: HeatTransferTopOptProblem{dim, T}

```
  T = T_left                         T = T_right
  ┌────────────────────────────────────────┐
  │                                        │
  │                                        │
  │          k(ρ)∇²T = 0                   │
  │         (heat conduction)              │
  │                                        │
  │                                        │
  └────────────────────────────────────────┘
            ▲ q (heat flux on boundary)
            │
  ┌────────────────────────────────────────┐
  │    ρ = design density (0 to 1)         │
  │    k(ρ) = penalized conductivity       │
  │    q = heat flux (NOT penalized)       │
  └────────────────────────────────────────┘
```

A steady-state heat conduction problem with:
- Temperature BCs: T = `T_left` on left boundary, T = `T_right` on right boundary
- Heat flux BCs: q on specified boundaries (facesets)
- Objective: minimize thermal compliance J = ∫ q·T dΓ


Constructor arguments:
- `nels`: tuple of number of elements in each dimension
- `sizes`: tuple of element sizes
- `k`: thermal conductivity (W/m·K)
- `Tleft`: temperature on left boundary
- `Tright`: temperature on right boundary
- `heatflux`: Dict mapping faceset names to heat flux values (W/m²)
  - Positive values = heat entering the domain (heat source on boundary)
  - Negative values = heat leaving the domain (heat sink on boundary)
- `cload`: Dict mapping a node index to a concentrated heat source value (W).
  Positive values inject heat at that node; negative values remove heat.
  This is the point-source analogue of the distributed `heatflux` and is the
  setup that produces the classic branching "conductivity tree" topology.
- `Tfix`: Dict mapping a node index to a prescribed temperature (K), applied
  as a point Dirichlet BC. Use it to pin the temperature at individual nodes
  (e.g. a point cold sink) instead of along a whole boundary face.

Note: Heat flux q and concentrated heat sources are NOT penalized in the
assembly. Only conductivity k(ρ) is penalized.
"""
struct HeatConductionProblem{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler,T},
    Th<:AbstractDict{String,T},
    Tc2<:AbstractDict{Int,T},
    Tm<:Metadata,
} <: HeatTransferTopOptProblem{dim,T}
    rect_grid::Tr
    k::T
    ch::Tc
    heatfluxdict::Th
    cloaddict::Tc2
    metadata::Tm
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::HeatConductionProblem)
    return println(io, "TopOpt heat conduction problem")
end

getheatfluxdict(p::HeatConductionProblem) = p.heatfluxdict
# Wrap each scalar heat source in a 1-element vector to match the structural
# `Dict{node => [values...]}` convention used by `make_cload`.
getcloaddict(p::HeatConductionProblem) = Dict(node => [v] for (node, v) in p.cloaddict)

"""
    HeatConductionProblem(nels, sizes, k=1.0; celltype=:Linear, Tleft=0.0, Tright=0.0, Ttop=nothing, Tbottom=nothing, heatflux=Dict{String,Float64}(), cload=Dict{Int,Float64}(), Tfix=Dict{Int,Float64}())

Create a 2D/3D heat conduction problem on a rectangular domain.

Temperature (Dirichlet) BCs default to `0.0` on the left and right
boundaries; pass `Tleft`/`Tright` to set them, or `nothing` to leave a
side free (no Dirichlet BC there). `Ttop` and `Tbottom` default to
`nothing` (free); set them to apply temperature BCs on the top/bottom.
`Tfix` applies point Dirichlet BCs at individual nodes.

Heat flux BCs can be applied on any faceset via the `heatflux` argument.
Point heat sources can be applied at nodes via the `cload` argument.

`celltype` is either `:Linear` or `:Quadratic` to determine the order of the
geometric and field basis functions and element type.

Example:
```julia
nels = (60, 20)
sizes = (1.0, 1.0)
k = 1.0
# Apply heat flux on top boundary (faceset "top")
heatflux = Dict("top" => 100.0)  # 100 W/m² into the domain
problem = HeatConductionProblem(nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux)

# Point heat source at the top-center node, cold bottom (classic tree setup):
nx, ny = nels
center_top_node = div(nx, 2) + 1 + ny * (nx + 1)
cload = Dict(center_top_node => 1.0)
problem = HeatConductionProblem(nels, sizes, k; Tleft=nothing, Tright=nothing, Tbottom=0.0, cload=cload)

# Point source top-center, point sink bottom-center (branching tree):
center_bottom_node = div(nx, 2) + 1
problem = HeatConductionProblem(nels, sizes, k;
    Tleft=nothing, Tright=nothing, Ttop=nothing, Tbottom=nothing,
    cload=Dict(center_top_node => 1.0),
    Tfix=Dict(center_bottom_node => 0.0),
)
```
"""
function HeatConductionProblem(
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    k=1.0;
    celltype::Symbol=:Linear,
    Tleft=0.0,
    Tright=0.0,
    Ttop=nothing,
    Tbottom=nothing,
    heatflux=Dict{String,Float64}(),
    cload=Dict{Int,Float64}(),
    Tfix=Dict{Int,Float64}(),
) where {dim}
    return _HeatConductionProblem(
        _celltype_tag(celltype),
        nels,
        sizes,
        k;
        Tleft=Tleft,
        Tright=Tright,
        Ttop=Ttop,
        Tbottom=Tbottom,
        heatflux=heatflux,
        cload=cload,
        Tfix=Tfix,
    )
end
function _HeatConductionProblem(
    ::Val{CellType},
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    k=1.0;
    Tleft=0.0,
    Tright=0.0,
    Ttop=nothing,
    Tbottom=nothing,
    heatflux=Dict{String,Float64}(),
    cload=Dict{Int,Float64}(),
    Tfix=Dict{Int,Float64}(),
) where {dim,CellType}
    # Promote the numeric BC values (skipping `nothing`) to pick the element
    # type, defaulting to Float64 when all BCs are `nothing`.
    bc_vals = [v for v in (Tleft, Tright, Ttop, Tbottom) if v !== nothing]
    T = float(
        promote_type(
            eltype(sizes),
            typeof(k),
            map(typeof, bc_vals)...,
            map(typeof, values(cload))...,
            map(typeof, values(Tfix))...,
            Float64,
        ),
    )

    if CellType === :Linear
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Linear)
    else
        rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Quadratic)
    end

    # Add boundary node sets
    if haskey(rect_grid.grid.nodesets, "left_boundary")
        pop!(rect_grid.grid.nodesets, "left_boundary")
    end
    addnodeset!(rect_grid.grid, "left_boundary", x -> left(rect_grid, x))

    if haskey(rect_grid.grid.nodesets, "right_boundary")
        pop!(rect_grid.grid.nodesets, "right_boundary")
    end
    addnodeset!(rect_grid.grid, "right_boundary", x -> right(rect_grid, x))

    if haskey(rect_grid.grid.nodesets, "top_boundary")
        pop!(rect_grid.grid.nodesets, "top_boundary")
    end
    addnodeset!(rect_grid.grid, "top_boundary", x -> top(rect_grid, x))

    if haskey(rect_grid.grid.nodesets, "bottom_boundary")
        pop!(rect_grid.grid.nodesets, "bottom_boundary")
    end
    addnodeset!(rect_grid.grid, "bottom_boundary", x -> bottom(rect_grid, x))

    # Create temperature field (scalar, 1 DOF per node)
    dh = DofHandler(rect_grid.grid)
    refshape = Ferrite.getrefshape(eltype(rect_grid.grid.cells))
    if CellType === :Linear
        ip = Lagrange{refshape,1}()
        add!(dh, :T, ip)  # Temperature is a scalar field
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :T, ip)
    end
    close!(dh)

    # Apply temperature boundary conditions. A side left at `nothing` has no
    # Dirichlet BC (free boundary); a numeric value fixes the temperature there.
    ch = ConstraintHandler(dh)
    if Tleft !== nothing
        add!(
            ch,
            Dirichlet(:T, getnodeset(rect_grid.grid, "left_boundary"), (x, t) -> T(Tleft)),
        )
    end
    if Tright !== nothing
        add!(
            ch,
            Dirichlet(
                :T, getnodeset(rect_grid.grid, "right_boundary"), (x, t) -> T(Tright)
            ),
        )
    end
    if Ttop !== nothing
        add!(
            ch, Dirichlet(:T, getnodeset(rect_grid.grid, "top_boundary"), (x, t) -> T(Ttop))
        )
    end
    if Tbottom !== nothing
        add!(
            ch,
            Dirichlet(
                :T, getnodeset(rect_grid.grid, "bottom_boundary"), (x, t) -> T(Tbottom)
            ),
        )
    end
    if !isempty(Tfix)
        # Prescribe the temperature at individual nodes (point Dirichlet BCs).
        fix_nodes = collect(keys(Tfix))
        # Ferrite's Dirichlet applies a single value to a node set; since each
        # fixed node can carry a different temperature, add one Dirichlet per
        # distinct value.
        for val in unique(values(Tfix))
            nodes = [n for n in fix_nodes if Tfix[n] == val]
            add!(ch, Dirichlet(:T, Set(nodes), (x, t) -> T(val)))
        end
    end
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    # Convert heatflux dict to proper type
    heatfluxdict = Dict{String,T}()
    for (key, val) in heatflux
        heatfluxdict[key] = T(val)
    end

    # Convert cload dict (node index => heat source value) to proper type.
    cloaddict = Dict{Int,T}()
    for (node, val) in cload
        cloaddict[node] = T(val)
    end

    return HeatConductionProblem(rect_grid, T(k), ch, heatfluxdict, cloaddict, metadata)
end

nnodespercell(p::HeatConductionProblem) = nnodespercell(p.rect_grid)

"""
    HeatTree(nels, sizes, k=1.0; celltype=:Linear, q=1.0)

Convenience constructor for the classic heat-conduction topology-optimization
benchmark ([BendsoeSigmund2003](@cite) §1.3, Fig. 1.4):
distributed heat flux `q` enters through the full top edge, the full bottom
edge is held at `T = 0` (cold sink), and the left/right sides are insulated
(free). Minimizing thermal compliance with this setup produces the branching
"conductivity tree" — a root structure at the cold bottom that branches and
tapers toward the hot top.

Arguments:
- `nels`: tuple of number of elements per dimension
- `sizes`: tuple of element sizes
- `k`: thermal conductivity
- `q`: heat flux on the top boundary (W/m², positive = into the domain)
- `celltype`: either `:Linear` or `:Quadratic`
"""
function HeatTree(
    nels::NTuple{dim,Int}, sizes::NTuple{dim}, k=1.0; celltype::Symbol=:Linear, q=1.0
) where {dim}
    return HeatConductionProblem(
        nels,
        sizes,
        k;
        celltype=celltype,
        Tleft=nothing,
        Tright=nothing,
        Ttop=nothing,
        Tbottom=0.0,
        heatflux=Dict("top" => q),
    )
end

"""
    WallWithOpening(nels::NTuple{2,Int}, sizes::NTuple{2}, E=1.0, ν=0.3, q=1.0e-2; celltype=:Linear, planestress=false, kσpoint=:quadrature)

The 2D wall-reinforcement benchmark of [FerrariSigmundGuest2021](@cite)
(Example 5.2): a square wall clamped at the base on four support patches,
loaded laterally by a total load `q` spread over the left edge. A void
opening (passive `x = 0`) occupies the upper-middle of the wall,
surrounded by a passive solid frame (`x = 1`) of thickness `Ly/40`,
matching the reference setup. The active domain is the remaining material.

```
     q →
   ┌───────┐
   │  ┌───┐│   opening (passive void, framed by passive solid)
   │  │   ││
   │  │   ││
   │  │   ││
   │  └───┘│
   │       │
   │       │
   └───────┘
   ▒ ▒   ▒ ▒  ← support patches (clamped, u_x = u_y = 0)
```

The passive regions are exposed through [`getpassive_solid`](@ref) and
[`getpassive_void`](@ref).
"""
struct WallWithOpening{T,N,M,Tr<:RectilinearGrid{2,T,N,M},Tc,Td,Tm} <:
       StiffnessTopOptProblem{2,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    cloads::Td
    metadata::Tm
    planestress::Bool
    kσpoint::Symbol
    passive_solid::Vector{Int}
    passive_void::Vector{Int}
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::WallWithOpening)
    return println(io, "TopOpt wall-with-opening problem")
end
getplanestress(p::WallWithOpening) = p.planestress
getkσpoint(p::WallWithOpening) = p.kσpoint
getpassive_solid(p::WallWithOpening) = p.passive_solid
getpassive_void(p::WallWithOpening) = p.passive_void
nnodespercell(p::WallWithOpening) = nnodespercell(p.rect_grid)
getcloaddict(p::WallWithOpening) = p.cloads

function WallWithOpening(
    nels::NTuple{2,Int},
    sizes::NTuple{2},
    E=1.0,
    ν=0.3,
    q=1.0e-2;
    celltype::Symbol=:Linear,
    planestress::Bool=false,
    kσpoint::Symbol=:quadrature,
)
    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(q)))
    rect_grid = RectilinearGrid(nels, T.(sizes); celltype=:Linear)
    nelx, nely = nels
    Lx = rect_grid.corners[2][1] - rect_grid.corners[1][1]
    Ly = rect_grid.corners[2][2] - rect_grid.corners[1][2]

    # support patches along the base (the topBuck250 columns 1:9, 120:128,
    # 257:265, 313:321 at 320 elements, scaled)
    patchcols = vcat(
        1:max(1, 9 * nelx ÷ 320),
        max(1, 120 * nelx ÷ 320):max(1, 128 * nelx ÷ 320),
        max(1, 257 * nelx ÷ 320):max(1, 265 * nelx ÷ 320),
        max(1, 313 * nelx ÷ 320):nelx,
    )
    if haskey(rect_grid.grid.nodesets, "supports")
        pop!(rect_grid.grid.nodesets, "supports")
    end
    # the base row nodes at the patch columns: node row 1 (y = 0), column ix
    base_nodes = Int[]
    for (i, node) in enumerate(Ferrite.getnodes(rect_grid.grid))
        c = Ferrite.get_node_coordinate(node)
        c[2] ≈ rect_grid.corners[1][2] || continue
        col = round(Int, c[1] / (Lx / nelx))
        col in patchcols || continue
        push!(base_nodes, i)
    end
    addnodeset!(rect_grid.grid, "supports", Set(base_nodes))
    if haskey(rect_grid.grid.nodesets, "wall_load")
        pop!(rect_grid.grid.nodesets, "wall_load")
    end
    addnodeset!(rect_grid.grid, "wall_load", x -> left(rect_grid, x))

    dh = DofHandler(rect_grid.grid)
    refshape = Ferrite.getrefshape(eltype(rect_grid.grid.cells))
    if celltype === :Linear
        ip = Lagrange{refshape,1}()
        add!(dh, :u, ip^2)
    else
        ip = Lagrange{refshape,2}()
        add!(dh, :u, ip^2)
    end
    close!(dh)
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(
        :u, getnodeset(rect_grid.grid, "supports"), (x, t) -> zeros(T, 2), collect(1:2)
    )
    add!(ch, dbc)
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    # lateral load over the whole left edge (trapezoidal, total q)
    loadnodes = sort(
        collect(getnodeset(rect_grid.grid, "wall_load"));
        by=ni -> get_node_coordinate(rect_grid.grid.nodes[ni])[2],
    )
    nseg = length(loadnodes)
    w = T(q / max(nseg - 1, 1))
    cloads = Dict{Int,Vector{T}}()
    for (k, ni) in enumerate(loadnodes)
        wi = (k == 1 || k == nseg) ? w / 2 : w
        cloads[ni] = [wi, zero(T)]
    end

    # passive regions (the topBuck250 code, scaled): solid frames and the
    # void opening. Rows count from the bottom (iy = 1 at y = 0); the
    # opening occupies rows 2nely/5:nely, columns 2nelx/5:nelx-nelx/5.
    solid = Int[]
    void = Int[]
    for ix in 1:nelx, iy in 1:nely
        e = (ix - 1) * nely + iy          # x-fastest cell numbering
        inop = (iy >= 2 * nely ÷ 5) && (ix >= 2 * nelx ÷ 5) && (ix <= nelx - nelx ÷ 5)
        inop && continue                 # the opening (void, added below)
        # frames
        bottom = iy <= nely ÷ 40
        side = ix <= nelx ÷ 40 || ix > nelx - nelx ÷ 40
        opl =
            (iy >= 2 * nely ÷ 5 - nely ÷ 40) &&
            (ix >= 2 * nelx ÷ 5 - nelx ÷ 40) &&
            (ix <= 2 * nelx ÷ 5 - 1)
        opr =
            (iy >= 2 * nely ÷ 5 - nely ÷ 40) &&
            (ix >= nelx - nelx ÷ 5 + 1) &&
            (ix <= nelx - nelx ÷ 5 + nelx ÷ 40)
        opb =
            (iy >= 2 * nely ÷ 5 - nely ÷ 40) &&
            (iy <= 2 * nely ÷ 5 - 1) &&
            (ix >= 2 * nelx ÷ 5) &&
            (ix <= nelx - nelx ÷ 5)
        (bottom || side || opl || opr || opb) && push!(solid, e)
    end
    for ix in 1:nelx, iy in 1:nely
        e = (ix - 1) * nely + iy
        (iy >= 2 * nely ÷ 5) &&
            (ix >= 2 * nelx ÷ 5) &&
            (ix <= nelx - nelx ÷ 5) &&
            push!(void, e)
    end
    return WallWithOpening(
        rect_grid, E, ν, ch, cloads, metadata, planestress, kσpoint, solid, void
    )
end
