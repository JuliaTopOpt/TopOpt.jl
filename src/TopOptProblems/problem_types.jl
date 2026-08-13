using Ferrite: Cell

"""
    abstract type StiffnessTopOptProblem{dim, T} <: AbstractTopOptProblem end

An abstract stiffness topology optimization problem. All subtypes must have the following fields:
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
"""
abstract type StiffnessTopOptProblem{dim,T} <: AbstractTopOptProblem end

# Fallbacks
getdim(::StiffnessTopOptProblem{dim,T}) where {dim,T} = dim
floattype(::StiffnessTopOptProblem{dim,T}) where {dim,T} = T
getE(p::StiffnessTopOptProblem) = p.E
getν(p::StiffnessTopOptProblem) = p.ν
getgeomorder(p::StiffnessTopOptProblem) = nnodespercell(p) in (9, 27) ? 2 : 1
getdensity(::StiffnessTopOptProblem{dim,T}) where {dim,T} = T(0)
getmetadata(p::StiffnessTopOptProblem) = p.metadata
getdh(p::StiffnessTopOptProblem) = p.ch.dh
getcloaddict(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,Vector{T}}()
getpressuredict(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,T}()
getfacesets(p::StiffnessTopOptProblem{dim,T}) where {dim,T} = Dict{String,Tuple{Int,T}}()
Ferrite.getncells(problem::StiffnessTopOptProblem) = Ferrite.getncells(getdh(problem).grid)

"""
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
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::PointLoadCantilever)
    return println(io, "TopOpt point load cantilever beam problem")
end

"""
    PointLoadCantilever(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force) where {dim, CellType}

- `dim`: dimension of the problem
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `nels`: number of elements in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `sizes`: the size of each element in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `CellType`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

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

problem = PointLoadCantilever(Val{celltype}, nels, sizes, E, ν, force)
```
"""
function PointLoadCantilever(
    ::Type{Val{CellType}},
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0,
) where {dim,CellType}
    iseven(nels[2]) && (length(nels) < 3 || iseven(nels[3])) ||
        throw("Grid does not have an even number of elements along the y and/or z axes.")

    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force)))
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    else
        rect_grid = RectilinearGrid(Val{:Quadratic}, nels, T.(sizes))
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

    return PointLoadCantilever(rect_grid, E, ν, ch, force, force_dof, metadata)
end

"""
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
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::HalfMBB)
    return println(io, "TopOpt half MBB problem")
end

"""
    HalfMBB(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E, ν, force) where {dim, CellType}

- `dim`: dimension of the problem
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the top left of half the MBB (positive is downward)
- `nels`: number of elements in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `sizes`: the size of each element in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `CellType`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

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

problem = HalfMBB(Val{celltype}, nels, sizes, E, ν, force)
```
"""
function HalfMBB(
    ::Type{Val{CellType}},
    nels::NTuple{dim,Int},
    sizes::NTuple{dim},
    E=1.0,
    ν=0.3,
    force=1.0,
) where {dim,CellType}
    T = float(promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force)))
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    else
        rect_grid = RectilinearGrid(Val{:Quadratic}, nels, T.(sizes))
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

    return HalfMBB(rect_grid, E, ν, ch, force, force_dof, metadata)
end

nnodespercell(p::Union{PointLoadCantilever,HalfMBB}) = nnodespercell(p.rect_grid)
function getcloaddict(p::Union{PointLoadCantilever{dim,T},HalfMBB{dim,T}}) where {dim,T}
    f = T[0, -p.force, 0]
    fnode = Tuple(getnodeset(p.rect_grid.grid, "down_force"))[1]
    return Dict{Int,Vector{T}}(fnode => f)
end

"""
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
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::LBeam)
    return println(io, "TopOpt L-beam problem")
end

"""
    LBeam(::Type{Val{CellType}}, ::Type{T}=Float64; length = 100, height = 100, upperslab = 50, lowerslab = 50, E = 1.0, ν = 0.3, force = 1.0) where {T, CellType}

- `T`: number type for computations and coordinates
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `length`, `height`, `upperslab` and `lowerslab` are explained in [`LGrid`](@ref).
- `CellType`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.

Example:
```
E = 1.0;
ν = 0.3;
force = 1.0;

# Linear elements and linear basis functions
celltype = :Linear

# Quadratic elements and quadratic basis functions
#celltype = :Quadratic

problem = LBeam(Val{celltype}, E = E, ν = ν, force = force)
```
"""
function LBeam(
    ::Type{Val{CellType}},
    (::Type{T})=Float64;
    length=100,
    height=100,
    upperslab=50,
    lowerslab=50,
    E=1.0,
    ν=0.3,
    force=1.0,
) where {T,CellType}
    # Create displacement field u
    grid = LGrid(
        Val{CellType},
        T;
        length=length,
        height=height,
        upperslab=upperslab,
        lowerslab=lowerslab,
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
        E, ν, ch, force, force_dof, metadata
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
    f = T[0, -p.force]
    fnode = Tuple(getnodeset(getdh(p).grid, "load"))[1]
    return Dict{Int,Vector{T}}(fnode => f)
end

"""
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
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::TieBeam)
    return println(io, "TopOpt tie-beam problem")
end

"""
    TieBeam(::Type{Val{CellType}}, ::Type{T} = Float64, refine = 1, force = T(1); E = T(1), ν = T(0.3)) where {T, CellType}

- `T`: number type for computations and coordinates
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `refine`: an integer value of 1 or greater that specifies the mesh refinement extent. A value of 1 gives the standard tie-beam problem in literature.
- `CellType`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.
"""
function TieBeam(
    ::Type{Val{CellType}}, (::Type{T})=Float64; refine=1, force=T(1), E=T(1), ν=T(0.3)
) where {T,CellType}
    grid = TieBeamGrid(Val{CellType}, T; refine=refine)

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
    return TieBeam{T,N,M,typeof(ch),typeof(metadata)}(E, ν, force, ch, metadata)
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
getk(p::HeatTransferTopOptProblem) = p.k
getmetadata(p::HeatTransferTopOptProblem) = p.metadata
getdh(p::HeatTransferTopOptProblem) = p.ch.dh
getpressuredict(p::HeatTransferTopOptProblem{dim,T}) where {dim,T} = Dict{String,T}()
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
    HeatConductionProblem(::Type{Val{CellType}}, nels, sizes, k=1.0; Tleft=0.0, Tright=0.0, Ttop=nothing, Tbottom=nothing, heatflux=Dict{String,Float64}(), cload=Dict{Int,Float64}(), Tfix=Dict{Int,Float64}())

Create a 2D/3D heat conduction problem on a rectangular domain.

Temperature (Dirichlet) BCs default to `0.0` on the left and right
boundaries; pass `Tleft`/`Tright` to set them, or `nothing` to leave a
side free (no Dirichlet BC there). `Ttop` and `Tbottom` default to
`nothing` (free); set them to apply temperature BCs on the top/bottom.
`Tfix` applies point Dirichlet BCs at individual nodes.

Heat flux BCs can be applied on any faceset via the `heatflux` argument.
Point heat sources can be applied at nodes via the `cload` argument.

Example:
```julia
nels = (60, 20)
sizes = (1.0, 1.0)
k = 1.0
# Apply heat flux on top boundary (faceset "top")
heatflux = Dict("top" => 100.0)  # 100 W/m² into the domain
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux)

# Point heat source at the top-center node, cold bottom (classic tree setup):
nx, ny = nels
center_top_node = div(nx, 2) + 1 + ny * (nx + 1)
cload = Dict(center_top_node => 1.0)
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k; Tleft=nothing, Tright=nothing, Tbottom=0.0, cload=cload)

# Point source top-center, point sink bottom-center (branching tree):
center_bottom_node = div(nx, 2) + 1
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k;
    Tleft=nothing, Tright=nothing, Ttop=nothing, Tbottom=nothing,
    cload=Dict(center_top_node => 1.0),
    Tfix=Dict(center_bottom_node => 0.0),
)
```
"""
function HeatConductionProblem(
    ::Type{Val{CellType}},
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
        rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    else
        rect_grid = RectilinearGrid(Val{:Quadratic}, nels, T.(sizes))
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
    HeatTree(::Type{Val{CellType}}, nels, sizes, k=1.0; q=1.0)

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
"""
function HeatTree(
    ::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, k=1.0; q=1.0
) where {dim,CellType}
    return HeatConductionProblem(
        Val{CellType},
        nels,
        sizes,
        k;
        Tleft=nothing,
        Tright=nothing,
        Ttop=nothing,
        Tbottom=0.0,
        heatflux=Dict("top" => q),
    )
end
