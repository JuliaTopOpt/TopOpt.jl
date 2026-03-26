using Ferrite: Cell

"""
    abstract type StiffnessTopOptProblem{dim, T} <: AbstractTopOptProblem end

An abstract stiffness topology optimization problem. All subtypes must have the following fields:
- `ch`: a `Ferrite.ConstraintHandler` struct
- `metadata`: Metadata having various cell-node-dof relationships
- `black`: a `BitVector` of length equal to the number of elements where `black[e]` is 1 iff the `e`^th element must be part of the final design
- `white`:  a `BitVector` of length equal to the number of elements where `white[e]` is 1 iff the `e`^th element must not be part of the final design
- `varind`: an `AbstractVector{Int}` of length equal to the number of elements where `varind[e]` gives the index of the decision variable corresponding to element `e`. Because some elements can be fixed to be black or white, not every element has a decision variable associated.
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
    ch::ConstraintHandler{<:DofHandler{dim, <:Cell{dim,N,M}, T}, T}
    force::T
    force_dof::Integer
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
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
- `black`: a `BitVector` of length equal to the number of elements where `black[e]` is 1 iff the `e`^th element must be part of the final design
- `white`:  a `BitVector` of length equal to the number of elements where `white[e]` is 1 iff the `e`^th element must not be part of the final design
- `varind`: an `AbstractVector{Int}` of length equal to the number of elements where `varind[e]` gives the index of the decision variable corresponding to element `e`. Because some elements can be fixed to be black or white, not every element has a decision variable associated.
"""
struct PointLoadCantilever{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler{dim,<:Cell{dim,N,M},T},T},
    Tf<:Integer,
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
} <: StiffnessTopOptProblem{dim,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::PointLoadCantilever)
    return println("TopOpt point load cantilever beam problem")
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

    _T = promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force))
    if _T <: Integer
        T = Float64
    else
        T = _T
    end
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    else
        rect_grid = RectilinearGrid(Val{:Quadratic}, nels, T.(sizes))
    end

    if haskey(rect_grid.grid.facesets, "fixed_all")
        pop!(rect_grid.grid.facesets, "fixed_all")
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
    if CellType === :Linear || dim === 3
        push!(dh, :u, dim) # Add a displacement field
    else
        ip = Lagrange{2,RefCube,2}()
        push!(dh, :u, dim, ip) # Add a displacement field        
    end
    close!(dh)

    ch = ConstraintHandler(dh)

    #dbc = Dirichlet(:u, getfaceset(rect_grid.grid, "fixed_all"), (x,t) -> zeros(T, dim), collect(1:dim))
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

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return PointLoadCantilever(
        rect_grid, E, ν, ch, force, force_dof, black, white, varind, metadata
    )
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
    ch::ConstraintHandler{<:DofHandler{dim, <:Cell{dim,N,M}, T}, T}
    force::T
    force_dof::Integer
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
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
- `black`: a `BitVector` of length equal to the number of elements where `black[e]` is 1 iff the `e`^th element must be part of the final design
- `white`:  a `BitVector` of length equal to the number of elements where `white[e]` is 1 iff the `e`^th element must not be part of the final design
- `varind`: an `AbstractVector{Int}` of length equal to the number of elements where `varind[e]` gives the index of the decision variable corresponding to element `e`. Because some elements can be fixed to be black or white, not every element has a decision variable associated.
"""
struct HalfMBB{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler{dim,<:Cell{dim,N,M},T},T},
    Tf<:Integer,
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
} <: StiffnessTopOptProblem{dim,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::HalfMBB)
    return println("TopOpt half MBB problem")
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
    _T = promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force))
    if _T <: Integer
        T = Float64
    else
        T = _T
    end
    if CellType === :Linear || dim === 3
        rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    else
        rect_grid = RectilinearGrid(Val{:Quadratic}, nels, T.(sizes))
    end

    if haskey(rect_grid.grid.facesets, "fixed_u1")
        pop!(rect_grid.grid.facesets, "fixed_u1")
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
    if CellType === :Linear || dim === 3
        push!(dh, :u, dim)
    else
        ip = Lagrange{2,RefCube,2}()
        push!(dh, :u, dim, ip)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    #dbc1 = Dirichlet(:u, getfaceset(rect_grid.grid, "fixed_u1"), (x,t)->T[0], [1])
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

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return HalfMBB(rect_grid, E, ν, ch, force, force_dof, black, white, varind, metadata)
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
    ch::ConstraintHandler{<:DofHandler{2, <:Cell{2,N,M}, T}, T}
    force::T
    force_dof::Integer
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
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
- `black`: a `BitVector` of length equal to the number of elements where `black[e]` is 1 iff the `e`^th element must be part of the final design
- `white`:  a `BitVector` of length equal to the number of elements where `white[e]` is 1 iff the `e`^th element must not be part of the final design
- `varind`: an `AbstractVector{Int}` of length equal to the number of elements where `varind[e]` gives the index of the decision variable corresponding to element `e`. Because some elements can be fixed to be black or white, not every element has a decision variable associated.
"""
struct LBeam{
    T,
    N,
    M,
    Tc<:ConstraintHandler{<:DofHandler{2,<:Cell{2,N,M},T},T},
    Tf<:Integer,
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
} <: StiffnessTopOptProblem{2,T}
    E::T
    ν::T
    ch::Tc
    force::T
    force_dof::Tf
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::LBeam) = println("TopOpt L-beam problem")

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
    if CellType === :Linear
        push!(dh, :u, 2)
    else
        ip = Lagrange{2,RefCube,2}()
        push!(dh, :u, 2, ip)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> T[0, 0], [1, 2])
    add!(ch, dbc)
    close!(ch)

    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    fnode = Tuple(getnodeset(grid, "load"))[1]
    node_dofs = metadata.node_dofs
    force_dof = node_dofs[2, fnode]

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    TInds = typeof(varind)
    TMeta = typeof(metadata)
    return LBeam(E, ν, ch, force, force_dof, black, white, varind, metadata)
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
    ch::ConstraintHandler{<:DofHandler{2, N, T, M}, T}
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
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
- `black`: a `BitVector` of length equal to the number of elements where `black[e]` is 1 iff the `e`^th element must be part of the final design
- `white`:  a `BitVector` of length equal to the number of elements where `white[e]` is 1 iff the `e`^th element must not be part of the final design
- `varind`: an `AbstractVector{Int}` of length equal to the number of elements where `varind[e]` gives the index of the decision variable corresponding to element `e`. Because some elements can be fixed to be black or white, not every element has a decision variable associated.
"""
struct TieBeam{
    T,
    N,
    M,
    Tc<:ConstraintHandler{<:DofHandler{2,<:Cell{2,N,M},T},T},
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
} <: StiffnessTopOptProblem{2,T}
    E::T
    ν::T
    force::T
    ch::Tc
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::TieBeam)
    return println("TopOpt tie-beam problem")
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
    ::Type{Val{CellType}}, (::Type{T})=Float64, refine=1, force=T(1); E=T(1), ν=T(0.3)
) where {T,CellType}
    grid = TieBeamGrid(Val{CellType}, T, refine)
    dh = DofHandler(grid)
    if CellType === :Linear
        push!(dh, :u, 2)
    else
        ip = Lagrange{2,RefCube,2}()
        push!(dh, :u, 2, ip)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "leftfixed"), (x, t) -> T[0, 0], [1, 2])
    add!(ch, dbc1)
    dbc2 = Dirichlet(:u, getfaceset(grid, "toproller"), (x, t) -> T[0], [2])
    add!(ch, dbc2)
    close!(ch)

    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return TieBeam(E, ν, force, ch, black, white, varind, metadata)
end

getdim(::TieBeam) = 2
nnodespercell(::TieBeam{T,N}) where {T,N} = N
function getpressuredict(p::TieBeam{T}) where {T}
    return Dict{String,T}("rightload" => 2 * p.force, "bottomload" => -p.force)
end
getfacesets(p::TieBeam) = getdh(p).grid.facesets

"""
```
    ******************************
    * Pin1         F1 ―――>       *
    *  o            |            * 
    *               v            *
    *  Pin2                      *
    *    o     F2 ―――>           *
    *           |                *
    *           v                *
    ******************************

    RayProblem(nels, pins, loads)

Constructs an instance of the type `RayProblem` that is a 2D beam with:
 - Number of elements `nels`, e.g. `(60, 20)` where each element is a 1 x 1 square,
 - Pinned locations `pins` where each pinned location is a `Vector` of length 2, e.g. `[[1, 18], [2, 8]]` indicating the locations of the pins, and
 - Loads specified in `loads` where `loads` is a dictionary mapping the location of each load to its vector value, e.g. `Dict([10, 18] => [1.0, -1.0], [5, 5] => [1.0, -1.0])` which defines a load of `[1.0, -1.0]` at the point located at `[10, 18]` and a similar load at the point located at `[5, 5]`.
```
"""
struct RayProblem{
    T,
    N,
    M,
    Tr<:RectilinearGrid{2,T,N,M},
    Tc<:ConstraintHandler{<:DofHandler{2,<:Cell{2,N,M},T},T},
    Tl<:Dict,
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
} <: StiffnessTopOptProblem{2,T}
    rect_grid::Tr
    E::T
    ν::T
    ch::Tc
    loads::Tl
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end
function RayProblem(
    nels::NTuple{2,Int}, pins::Vector{<:Vector}, loads::Dict{<:Vector,<:Vector}
)
    T = Float64
    rect_grid = RectilinearGrid(Val{:Linear}, nels, (1.0, 1.0))
    dim = length(nels)

    for (i, pin) in enumerate(pins)
        if haskey(rect_grid.grid.nodesets, "fixed$i")
            pop!(rect_grid.grid.nodesets, "fixed$i")
        end
        addnodeset!(rect_grid.grid, "fixed$i", x -> x ≈ pin)
    end
    for (i, k) in enumerate(keys(loads))
        if haskey(rect_grid.grid.nodesets, "force$i")
            pop!(rect_grid.grid.nodesets, "force$i")
        end
        addnodeset!(rect_grid.grid, "force$i", x -> x ≈ k)
    end

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    ip = Lagrange{2,RefCube,1}()
    push!(dh, :u, dim, ip) # Add a displacement field        
    close!(dh)

    ch = ConstraintHandler(dh)

    for i in 1:length(pins)
        dbc = Dirichlet(
            :u,
            getnodeset(rect_grid.grid, "fixed$i"),
            (x, t) -> zeros(T, dim),
            collect(1:dim),
        )
        add!(ch, dbc)
    end
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)
    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    loadsdict = Dict{Int,Vector{Float64}}(
        map(enumerate(keys(loads))) do (i, k)
            fnode = Tuple(getnodeset(rect_grid.grid, "force$i"))[1]
            (fnode => loads[k])
        end,
    )

    return RayProblem(rect_grid, 1.0, 0.3, ch, loadsdict, black, white, varind, metadata)
end
nnodespercell(p::RayProblem) = nnodespercell(p.rect_grid)
getcloaddict(p::RayProblem) = p.loads

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

All subtypes must have:
- `ch`: ConstraintHandler with temperature DOFs (1 DOF per node)
- `metadata`: Metadata with cell-node-dof relationships
- `black`, `white`, `varind`: Design variable management
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
getfacesets(p::HeatTransferTopOptProblem) = getdh(p).grid.facesets
Ferrite.getncells(problem::HeatTransferTopOptProblem) = Ferrite.getncells(getdh(problem).grid)
getgeomorder(p::HeatTransferTopOptProblem) = nnodespercell(p) in (9, 27) ? 2 : 1
getcloaddict(p::HeatTransferTopOptProblem{dim,T}) where {dim,T} = Dict{String,Vector{T}}()

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

Note: Heat flux q is NOT penalized in the assembly. Only conductivity k(ρ) is penalized.
"""
struct HeatConductionProblem{
    dim,
    T,
    N,
    M,
    Tr<:RectilinearGrid{dim, T, N, M},
    Tc<:ConstraintHandler{<:DofHandler{dim, <:Cell{dim, N, M}, T}, T},
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm<:Metadata,
    Th<:AbstractDict{String,T},
} <: HeatTransferTopOptProblem{dim, T}
    rect_grid::Tr
    k::T
    ch::Tc
    heatfluxdict::Th
    black::Tb
    white::Tw
    varind::Tv
    metadata::Tm
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::HeatConductionProblem)
    return println("TopOpt heat conduction problem")
end

getheatfluxdict(p::HeatConductionProblem) = p.heatfluxdict

"""
    HeatConductionProblem(::Type{Val{CellType}}, nels, sizes, k=1.0; Tleft=0.0, Tright=0.0, heatflux=Dict{String,Float64}())

Create a 2D/3D heat conduction problem on a rectangular domain.

Temperature BCs are applied on left (Tleft) and right (Tright) boundaries.
Heat flux BCs can be applied on any faceset via the `heatflux` argument.

Example:
```julia
nels = (60, 20)
sizes = (1.0, 1.0)
k = 1.0
# Apply heat flux on top boundary (faceset "top")
heatflux = Dict("top" => 100.0)  # 100 W/m² into the domain
problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux)
```
"""
function HeatConductionProblem(
    ::Type{Val{CellType}},
    nels::NTuple{dim, Int},
    sizes::NTuple{dim},
    k=1.0;
    Tleft=0.0,
    Tright=0.0,
    heatflux=Dict{String,Float64}(),
) where {dim, CellType}
    _T = promote_type(eltype(sizes), typeof(k), typeof(Tleft), typeof(Tright))
    if _T <: Integer
        T = Float64
    else
        T = _T
    end

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

    # Create temperature field (scalar, 1 DOF per node)
    dh = DofHandler(rect_grid.grid)
    if CellType === :Linear
        push!(dh, :T, 1)  # Temperature is a scalar field
    else
        ip = Lagrange{dim, RefCube, 2}()
        push!(dh, :T, 1, ip)
    end
    close!(dh)

    # Apply temperature boundary conditions
    ch = ConstraintHandler(dh)
    dbc_left = Dirichlet(:T, getnodeset(rect_grid.grid, "left_boundary"), (x, t) -> Tleft)
    dbc_right = Dirichlet(:T, getnodeset(rect_grid.grid, "right_boundary"), (x, t) -> Tright)
    add!(ch, dbc_left)
    add!(ch, dbc_right)
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    # Convert heatflux dict to proper type
    heatfluxdict = Dict{String,T}()
    for (key, val) in heatflux
        heatfluxdict[key] = T(val)
    end

    return HeatConductionProblem(
        rect_grid, T(k), ch, heatfluxdict, black, white, varind, metadata
    )
end

nnodespercell(p::HeatConductionProblem) = nnodespercell(p.rect_grid)
