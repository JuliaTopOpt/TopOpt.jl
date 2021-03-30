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
abstract type StiffnessTopOptProblem{dim, T} <: AbstractTopOptProblem end

# Fallbacks
getdim(::StiffnessTopOptProblem{dim, T}) where {dim, T} = dim
floattype(::StiffnessTopOptProblem{dim, T}) where {dim, T} = T
getE(p::StiffnessTopOptProblem) = p.E
getν(p::StiffnessTopOptProblem) = p.ν
getgeomorder(p::StiffnessTopOptProblem) = nnodespercell(p) == 9 ? 2 : 1
getdensity(::StiffnessTopOptProblem{dim, T}) where {dim, T} = T(0)
getmetadata(p::StiffnessTopOptProblem) = p.metadata
getdh(p::StiffnessTopOptProblem) = p.ch.dh
getcloaddict(p::StiffnessTopOptProblem{dim, T}) where {dim, T} = Dict{String, Vector{T}}()
getpressuredict(p::StiffnessTopOptProblem{dim, T}) where {dim, T} = Dict{String, T}()
getfacesets(p::StiffnessTopOptProblem{dim, T}) where {dim, T} = Dict{String, Tuple{Int, T}}()
Ferrite.getncells(problem::StiffnessTopOptProblem) = Ferrite.getncells(getdh(problem).grid)


"""
```
///**********************************
///*                                *
///*                                * |
///*                                * |
///********************************** v


@params struct PointLoadCantilever{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
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
@params struct PointLoadCantilever{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
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
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::PointLoadCantilever) = println("TopOpt point load cantilever beam problem")

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
function PointLoadCantilever(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E = 1.0, ν = 0.3, force = 1.0) where {dim, CellType}
    iseven(nels[2]) && (length(nels) < 3 || iseven(nels[3])) || throw("Grid does not have an even number of elements along the y and/or z axes.")

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
    addnodeset!(rect_grid.grid, "fixed_all", x -> left(rect_grid, x));
    
    if haskey(rect_grid.grid.nodesets, "down_force") 
        pop!(rect_grid.grid.nodesets, "down_force")
    end
    addnodeset!(rect_grid.grid, "down_force", x -> right(rect_grid, x) && middley(rect_grid, x));

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    if CellType === :Linear || dim === 3
        push!(dh, :u, dim) # Add a displacement field
    else
        ip = Lagrange{2, RefCube, 2}()
        push!(dh, :u, dim, ip) # Add a displacement field        
    end
    close!(dh)
    
    ch = ConstraintHandler(dh)

    #dbc = Dirichlet(:u, getfaceset(rect_grid.grid, "fixed_all"), (x,t) -> zeros(T, dim), collect(1:dim))
    dbc = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed_all"), (x,t) -> zeros(T, dim), collect(1:dim))
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
    
    return PointLoadCantilever(rect_grid, E, ν, ch, force, force_dof, black, white, varind, metadata)
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
@params struct HalfMBB{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
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
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::HalfMBB) = println("TopOpt half MBB problem")

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
function HalfMBB(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, E = 1.0, ν = 0.3, force = 1.0) where {dim, CellType}
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
    addnodeset!(rect_grid.grid, "fixed_u1", x -> left(rect_grid, x));
    
    if haskey(rect_grid.grid.nodesets, "fixed_u2")
        pop!(rect_grid.grid.nodesets, "fixed_u2")
    end
    addnodeset!(rect_grid.grid, "fixed_u2", x -> bottom(rect_grid, x) && right(rect_grid, x));

    if haskey(rect_grid.grid.nodesets, "down_force")
        pop!(rect_grid.grid.nodesets, "down_force")
    end
    addnodeset!(rect_grid.grid, "down_force", x -> top(rect_grid, x) && left(rect_grid, x));

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    if CellType === :Linear || dim === 3
        push!(dh, :u, dim)
    else
        ip = Lagrange{2, RefCube, 2}()
        push!(dh, :u, dim, ip)
    end
    close!(dh)
    
    ch = ConstraintHandler(dh)
    #dbc1 = Dirichlet(:u, getfaceset(rect_grid.grid, "fixed_u1"), (x,t)->T[0], [1])
    dbc1 = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed_u1"), (x,t)->T[0], [1])
    add!(ch, dbc1)
    dbc2 = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed_u2"), (x,t)->T[0], [2])
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

nnodespercell(p::Union{PointLoadCantilever, HalfMBB}) = nnodespercell(p.rect_grid)
function getcloaddict(p::Union{PointLoadCantilever{dim, T}, HalfMBB{dim, T}}) where {dim, T}
    f = T[0, -p.force, 0]
    fnode = Tuple(getnodeset(p.rect_grid.grid, "down_force"))[1]
    return Dict{Int, Vector{T}}(fnode => f)
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
@params struct LBeam{T, N, M} <: StiffnessTopOptProblem{2, T}
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
function LBeam(::Type{Val{CellType}}, ::Type{T}=Float64; length = 100, height = 100, upperslab = 50, lowerslab = 50, E = 1.0, ν = 0.3, force = 1.0) where {T, CellType}
    # Create displacement field u
    grid = LGrid(Val{CellType}, T, length=length, height=height, upperslab=upperslab, 
        lowerslab=lowerslab)

    dh = DofHandler(grid)
    if CellType === :Linear
        push!(dh, :u, 2)
    else
        ip = Lagrange{2, RefCube, 2}()
        push!(dh, :u, 2, ip)
    end
    close!(dh)
    
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfaceset(grid, "top"), (x,t)->T[0, 0], [1, 2])
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

function boundingbox(grid::Ferrite.Grid{dim}) where dim
    xmin1 = minimum(n->n.x[1], grid.nodes)
    xmax1 = maximum(n->n.x[1], grid.nodes)
    xmin2 = minimum(n->n.x[2], grid.nodes)
    xmax2 = maximum(n->n.x[2], grid.nodes)
    if dim === 2
        return ((xmin1, xmin2), (xmax1, xmax2))
    else
        xmin3 = minimum(n->n.x[3], grid.nodes)
        xmax3 = maximum(n->n.x[3], grid.nodes)
        return ((xmin1, xmin2, xmin3), (xmax1, xmax2, xmax3))
    end
end

function RectilinearTopology(b, topology = ones(getncells(getdh(b).grid)))
    bb = boundingbox(getdh(b).grid)
    go = getgeomorder(b)
    nels = Int.(round.(bb[2] .- bb[1]))
    dim = length(nels)
    if go === 1
        rectgrid = generate_grid(Quadrilateral, nels, Vec{dim}(bb[1]), Vec{dim}(bb[2]))
    elseif go === 2
        rectgrid = generate_grid(QuadraticQuadrilateral, nels, Vec{dim}(bb[1]), Vec{dim}(bb[2]))
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

nnodespercell(p::LBeam{T, N}) where {T, N} = N
getdim(::LBeam) = 2
function getcloaddict(p::LBeam{T}) where {T}
    f = T[0, -p.force]
    fnode = Tuple(getnodeset(getdh(p).grid, "load"))[1]
    return Dict{Int, Vector{T}}(fnode => f)
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
@params struct TieBeam{T, N, M} <: StiffnessTopOptProblem{2, T}
    E::T
    ν::T
    force::T
    ch::ConstraintHandler{<:DofHandler{2, <:Cell{2,N,M}, T}, T}
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
    metadata::Metadata
end
Base.show(::IO, ::MIME{Symbol("text/plain")}, ::TieBeam) = println("TopOpt tie-beam problem")

"""
    TieBeam(::Type{Val{CellType}}, ::Type{T} = Float64, refine = 1, force = T(1); E = T(1), ν = T(0.3)) where {T, CellType}

- `T`: number type for computations and coordinates
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `refine`: an integer value of 1 or greater that specifies the mesh refinement extent. A value of 1 gives the standard tie-beam problem in literature.
- `CellType`: can be either `:Linear` or `:Quadratic` to determine the order of the geometric and field basis functions and element type. Only isoparametric elements are supported for now.
"""
function TieBeam(::Type{Val{CellType}}, ::Type{T} = Float64, refine = 1, force = T(1); E = T(1), ν = T(0.3)) where {T, CellType}
    grid = TieBeamGrid(Val{CellType}, T, refine)
    dh = DofHandler(grid)
    if CellType === :Linear
        push!(dh, :u, 2)
    else
        ip = Lagrange{2, RefCube, 2}()
        push!(dh, :u, 2, ip)
    end
    close!(dh)
        
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "leftfixed"), (x,t)->T[0, 0], [1, 2])
    add!(ch, dbc1)
    dbc2 = Dirichlet(:u, getfaceset(grid, "toproller"), (x,t)->T[0], [2])
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
nnodespercell(::TieBeam{T, N}) where {T, N} = N
getpressuredict(p::TieBeam{T}) where {T} = Dict{String, T}("rightload"=>2*p.force, "bottomload"=>-p.force)
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
@params struct RayProblem{T, N, M} <: StiffnessTopOptProblem{2, T}
    rect_grid::RectilinearGrid{2, T, N, M}
    E::T
    ν::T
    ch::ConstraintHandler{<:DofHandler{2, <:Cell{2, N, M}, T}, T}
    loads::Dict
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
    metadata::Metadata
end
function RayProblem(nels::NTuple{2, Int}, pins::Vector{<:Vector}, loads::Dict{<:Vector, <:Vector})
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
        addnodeset!(rect_grid.grid, "force$i", x -> x ≈ k);
    end

    # Create displacement field u
    dh = DofHandler(rect_grid.grid)
    ip = Lagrange{2, RefCube, 1}()
    push!(dh, :u, dim, ip) # Add a displacement field        
    close!(dh)

    ch = ConstraintHandler(dh)

    for i in 1:length(pins)
        dbc = Dirichlet(:u, getnodeset(rect_grid.grid, "fixed$i"), (x,t) -> zeros(T, dim), collect(1:dim))
        add!(ch, dbc)
    end
    close!(ch)
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)
    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)
    
    loadsdict = Dict{Int, Vector{Float64}}(map(enumerate(keys(loads))) do (i, k)
        fnode = Tuple(getnodeset(rect_grid.grid, "force$i"))[1]
        (fnode => loads[k])
    end)

    return RayProblem(rect_grid, 1.0, 0.3, ch, loadsdict, black, white, varind, metadata)
end
nnodespercell(p::RayProblem) = nnodespercell(p.rect_grid)
getcloaddict(p::RayProblem) = p.loads
