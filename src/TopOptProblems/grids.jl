abstract type AbstractGrid{dim, T} end

const Vec = Ferrite.Vec

"""
```
struct RectilinearGrid{dim, T, N, M, TG<:Ferrite.Grid{dim, <:Ferrite.Cell{dim,N,M}, T}} <: AbstractGrid{dim, T}
    grid::TG
    nels::NTuple{dim, Int}
    sizes::NTuple{dim, T}
    corners::NTuple{2, Vec{dim, T}}
    white_cells::BitVector
    black_cells::BitVector
    constant_cells::BitVector
end
```

A type that represents a rectilinear grid with corner points `corners`.

- `dim`: dimension of the problem
- `T`: number type for computations and coordinates
- `N`: number of nodes in a cell of the grid
- `M`: number of faces in a cell of the grid
- `grid`: a Ferrite.Grid struct
- `nels`: number of elements in every dimension
- `sizes`: dimensions of each rectilinear cell
- `corners`: 2 corner points of the rectilinear grid
- `white_cells`: cells fixed to be void during optimization
- `black_cells`: cells fixed to have material during optimization
- `constant_cells`: cells fixed to be either void or have material during optimization
"""
@params struct RectilinearGrid{dim, T, N, M, TG<:Ferrite.Grid{dim, <:Ferrite.Cell{dim,N,M}, T}} <: AbstractGrid{dim, T}
    grid::TG
    nels::NTuple{dim, Int}
    sizes::NTuple{dim, T}
    corners::NTuple{2, Vec{dim,T}}
    white_cells::BitVector
    black_cells::BitVector
    constant_cells::BitVector
end

"""
    RectilinearGrid(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim,T}) where {dim, T, CellType}

Constructs an instance of [`RectilinearGrid`](@ref).

- `dim`: dimension of the problem
- `T`: number type for coordinates
- `nels`: number of elements in every dimension
- `sizes`: dimensions of each rectilinear cell

Example:

```
rectgrid = RectilinearGrid((60,20), (1.0,1.0))
```
"""
function RectilinearGrid(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim,T}) where {dim, T, CellType}
    if dim === 2
        if CellType === :Linear
            geoshape = Quadrilateral
        else
            geoshape = QuadraticQuadrilateral
        end
    else
        geoshape = Hexahedron
    end
    corner1 = Vec{dim}(fill(T(0), dim))
    corner2 = Vec{dim}((nels .* sizes))
    grid = generate_grid(geoshape, nels, corner1, corner2);

    N = nnodes(geoshape)
    M = Ferrite.nfaces(geoshape)
    ncells = prod(nels)
    return RectilinearGrid(grid, nels, sizes, (corner1, corner2), falses(ncells), falses(ncells), falses(ncells))
end

nnodespercell(::RectilinearGrid{dim,T,N,M}) where {dim, T, N, M} = N
nfacespercell(::RectilinearGrid{dim,T,N,M}) where {dim, T, N, M} = M

left(rectgrid::RectilinearGrid, x) = x[1] ≈ rectgrid.corners[1][1]
right(rectgrid::RectilinearGrid, x) = x[1] ≈ rectgrid.corners[2][1]
bottom(rectgrid::RectilinearGrid, x) = x[2] ≈ rectgrid.corners[1][2]
top(rectgrid::RectilinearGrid, x) = x[2] ≈ rectgrid.corners[2][2]
back(rectgrid::RectilinearGrid, x) = x[3] ≈ rectgrid.corners[1][3]
front(rectgrid::RectilinearGrid, x) = x[3] ≈ rectgrid.corners[2][3]
middlex(rectgrid::RectilinearGrid, x) = x[1] ≈ (rectgrid.corners[1][1] + rectgrid.corners[2][1]) / 2
middley(rectgrid::RectilinearGrid, x) = x[2] ≈ (rectgrid.corners[1][2] + rectgrid.corners[2][2]) / 2
middlez(rectgrid::RectilinearGrid, x) = x[3] ≈ (rectgrid.corners[1][3] + rectgrid.corners[2][3]) / 2

nnodes(cell::Type{Ferrite.Cell{dim,N,M}}) where {dim, N, M} = N
nnodes(cell::Ferrite.Cell) = nnodes(typeof(cell))

"""
    LGrid(::Type{Val{CellType}}, ::Type{T}; length = 100, height = 100, upperslab = 50, lowerslab = 50) where {T, CellType}
    LGrid(::Type{Val{CellType}}, nel1::NTuple{2,Int}, nel2::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T}, MR::Vec{2,T}) where {CellType, T}

Constructs a `Ferrite.Grid` that represents the following L-shaped grid.

```
        upperslab   UR
       ............
       .          .
       .          .
       .          . 
height .          .                     MR
       .          ......................
       .                               .
       .                               . lowerslab
       .                               .
       .................................
     LL             length


```

Examples:

```
LGrid(upperslab = 30, lowerslab = 70)
LGrid(Val{:Linear}, (2, 4), (2, 2), Vec{2,Float64}((0.0,0.0)), Vec{2,Float64}((2.0, 4.0)), Vec{2,Float64}((4.0, 2.0)))
```
"""
function LGrid(::Type{Val{CellType}}, ::Type{T}; length = 100, height = 100, upperslab = 50, lowerslab = 50) where {T, CellType}
    @assert length > upperslab
    @assert height > lowerslab
    LGrid(Val{CellType}, (upperslab, height), (length-upperslab, lowerslab), 
        Vec{2,T}((0.0,0.0)), Vec{2,T}((T(upperslab), T(height))), 
        Vec{2,T}((T(length), T(lowerslab))))
end
function LGrid(::Type{Val{CellType}}, nel1::NTuple{2,Int}, nel2::NTuple{2,Int}, 
    LL::Vec{2,T}, UR::Vec{2,T}, MR::Vec{2,T}) where {CellType, T}

    if CellType === :Linear
        return _LinearLGrid(nel1, nel2, LL, UR, MR)
    else
        return _QuadraticLGrid(nel1, nel2, LL, UR, MR)
    end
end

function _LinearLGrid(nel1::NTuple{2,Int}, nel2::NTuple{2,Int}, 
    LL::Vec{2,T}, UR::Vec{2,T}, MR::Vec{2,T}) where {T}

    @assert nel1[2] > nel2[2]

    midpointindy = round(Int, nel2[2]/2) + 1
    nodes = Node{2,T}[]
    cells = Quadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["right"]  = Set{Tuple{Int,Int}}()
    facesets["top"]    = Set{Tuple{Int,Int}}()
    nodesets = Dict{String, Set{Int}}()
    nodesets["load"] = Set{Int}()
    
    # Lower left rectangle
    nel_x1 = nel1[1]; nel_y1 = nel2[2];
    n_nodes_x1 = nel_x1 + 1; n_nodes_y1 = nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    _LR = Vec{2,T}((UR[1], LL[2]))
    _UL = Vec{2,T}((LL[1], MR[2]))
    _UR = Vec{2,T}((UR[1], MR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, _LR, _UR, _UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(cells, Quadrilateral((node_array1[i,j], node_array1[i+1,j], node_array1[i+1,j+1], node_array1[i,j+1])))
        if i == 1
            push!(boundary, (length(cells), 4))
        end
        if j == 1
            push!(boundary, (length(cells), 1))
        end
    end
    
    # Lower right rectangle
    offsetstep = (MR[1] - _LR[1])/nel2[1]
    indexoffset = length(nodes)

    nel_x2 = nel2[1] - 1; nel_y2 = nel2[2]
    n_nodes_x2 = nel_x2 + 1; n_nodes_y2 = nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2

    _LL = Vec{2,T}((_LR[1] + offsetstep, _LR[2]))
    _LR = Vec{2,T}((MR[1], LL[2]))
    _UL = Vec{2,T}((_UR[1] + offsetstep, MR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, _LL, _LR, MR, _UL)

    node_array2 = reshape(collect(indexoffset+1:indexoffset+n_nodes2), (n_nodes_x2, n_nodes_y2))
    for j in 1:nel_y2
        push!(cells, Quadrilateral((node_array1[end,j], node_array2[1,j], node_array2[1,j+1], node_array1[end,j+1])))
        j == 1 && push!(boundary, (length(cells), 1))
        j == nel_y2 && push!(boundary, (length(cells), 3))
        if nel_x2 == 1
            push!(boundary, (length(cells), 2))
            push!(facesets["right"], (length(cells), 2))
        end
        for i in 1:nel_x2
            push!(cells, Quadrilateral((node_array2[i,j], node_array2[i+1,j], node_array2[i+1,j+1], node_array2[i,j+1])))
            if i == nel_x2
                push!(boundary, (length(cells), 2))
                push!(facesets["right"], (length(cells), 2))
            end
            j == 1 && push!(boundary, (length(cells), 1))
            j == nel_y2 && push!(boundary, (length(cells), 3))
        end
    end

    push!(nodesets["load"], node_array2[end, midpointindy])

    # Upper left rectangle
    offsetstep = (UR[2] - MR[2])/(nel1[2] - nel2[2])
    indexoffset = length(nodes)

    nel_x3 = nel1[1]; nel_y3 = nel1[2] - nel2[2] - 1
    n_nodes_x3 = nel_x3 + 1; n_nodes_y3 = nel_y3 + 1
    n_nodes3 = n_nodes_x3 * n_nodes_y3

    _LL = Vec{2,T}((LL[1], MR[2] + offsetstep))
    _LR = Vec{2,T}((UR[1], MR[2] + offsetstep))
    _UL = Vec{2,T}((LL[1], UR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x3, n_nodes_y3, _LL, _LR, UR, _UL)

    # Generate cells
    node_array3 = reshape(collect(indexoffset+1:indexoffset+n_nodes3), (n_nodes_x3, n_nodes_y3))

    for i in 1:nel_x3
        push!(cells, Quadrilateral((node_array1[i,end], node_array1[i+1,end], node_array3[i+1,1], node_array3[i,1])))
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
    end
    for j in 1:nel_y3, i in 1:nel_x3
        push!(cells, Quadrilateral((node_array3[i,j], node_array3[i+1,j], node_array3[i+1,j+1], node_array3[i,j+1])))
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
        if j == nel_y3
            push!(boundary, (length(cells), 3))
            push!(facesets["top"], (length(cells), 3))
        end
    end
    
    boundary_matrix = Ferrite.boundaries_to_sparse(boundary)

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets, 
        boundary_matrix=boundary_matrix)
end

function _QuadraticLGrid(nel1::NTuple{2,Int}, nel2::NTuple{2,Int}, 
    LL::Vec{2,T}, UR::Vec{2,T}, MR::Vec{2,T}) where {T}

    @assert nel1[2] > nel2[2]

    midpointindy = round(Int, nel2[2]/2) + 1
    nodes = Node{2,T}[]
    cells = QuadraticQuadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["right"]  = Set{Tuple{Int,Int}}()
    facesets["top"]    = Set{Tuple{Int,Int}}()
    nodesets = Dict{String, Set{Int}}()
    nodesets["load"] = Set{Int}()
    
    # Lower left rectangle
    nel_x1 = nel1[1]; nel_y1 = nel2[2];
    n_nodes_x1 = 2*nel_x1 + 1; n_nodes_y1 = 2*nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    _LR = Vec{2,T}((UR[1], LL[2]))
    _UL = Vec{2,T}((LL[1], MR[2]))
    _UR = Vec{2,T}((UR[1], MR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, _LR, _UR, _UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(cells, QuadraticQuadrilateral((node_array1[2*i-1,2*j-1], node_array1[2*i+1,2*j-1], 
                                             node_array1[2*i+1,2*j+1],node_array1[2*i-1,2*j+1], 
                                             node_array1[2*i,2*j-1],node_array1[2*i+1,2*j], 
                                             node_array1[2*i,2*j+1],node_array1[2*i-1,2*j], 
                                             node_array1[2*i,2*j])))
        if i == 1
            push!(boundary, (length(cells), 4))
        end
        if j == 1
            push!(boundary, (length(cells), 1))
        end
    end
    
    # Lower right rectangle
    offsetstep = (MR[1] - _LR[1])/nel2[1]/2
    indexoffset = length(nodes)

    nel_x2 = nel2[1] - 1; nel_y2 = nel2[2]
    n_nodes_x2 = 2*nel_x2 + 2; n_nodes_y2 = 2*nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2

    _LL = Vec{2,T}((_LR[1] + offsetstep, _LR[2]))
    _LR = Vec{2,T}((MR[1], LL[2]))
    _UL = Vec{2,T}((_UR[1] + offsetstep, MR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, _LL, _LR, MR, _UL)

    node_array2 = reshape(collect(indexoffset+1:indexoffset+n_nodes2), (n_nodes_x2, n_nodes_y2))
    for j in 1:nel_y2
        push!(cells, QuadraticQuadrilateral((node_array1[end,2*j-1], node_array2[2,2*j-1], 
                                             node_array2[2,2*j+1], node_array1[end,2*j+1], 
                                             node_array2[1,2*j-1], node_array2[2,2*j], 
                                             node_array2[1,2*j+1], node_array1[end,2*j], 
                                             node_array2[1,2*j])))
        j == 1 && push!(boundary, (length(cells), 1))
        j == nel_y2 && push!(boundary, (length(cells), 3))
        if nel_x2 == 1
            push!(boundary, (length(cells), 2))
            push!(facesets["right"], (length(cells), 2))
        end
        for i in 1:nel_x2
            push!(cells, QuadraticQuadrilateral((node_array2[2*i,2*j-1], 
                                                 node_array2[2*i+2,2*j-1], 
                                                 node_array2[2*i+2,2*j+1], 
                                                 node_array2[2*i,2*j+1], 
                                                 node_array2[2*i+1,2*j-1], 
                                                 node_array2[2*i+2,2*j], 
                                                 node_array2[2*i+1,2*j+1], 
                                                 node_array2[2*i,2*j], 
                                                 node_array2[2*i+1,2*j])))
            if i == nel_x2
                push!(boundary, (length(cells), 2))
                push!(facesets["right"], (length(cells), 2))
            end
            j == 1 && push!(boundary, (length(cells), 1))
            j == nel_y2 && push!(boundary, (length(cells), 3))
        end
    end

    push!(nodesets["load"], node_array2[end, midpointindy])

    # Upper left rectangle
    offsetstep = (UR[2] - MR[2])/(nel1[2] - nel2[2])/2
    indexoffset = length(nodes)

    nel_x3 = nel1[1]; nel_y3 = nel1[2] - nel2[2] - 1
    n_nodes_x3 = 2*nel_x3 + 1; n_nodes_y3 = 2*nel_y3 + 2
    n_nodes3 = n_nodes_x3 * n_nodes_y3

    _LL = Vec{2,T}((LL[1], MR[2] + offsetstep))
    _LR = Vec{2,T}((UR[1], MR[2] + offsetstep))
    _UL = Vec{2,T}((LL[1], UR[2]))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x3, n_nodes_y3, _LL, _LR, UR, _UL)

    # Generate cells
    node_array3 = reshape(collect(indexoffset+1:indexoffset+n_nodes3), (n_nodes_x3, n_nodes_y3))

    for i in 1:nel_x3
        push!(cells, QuadraticQuadrilateral((node_array1[2i-1,end], node_array1[2i+1,end], 
                                             node_array3[2i+1,2], node_array3[2i-1,2], 
                                             node_array1[2i,end], node_array3[2i+1,1], 
                                             node_array3[2i,2], node_array3[2i-1,1],
                                             node_array3[2i,1])))
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
    end
    for j in 1:nel_y3, i in 1:nel_x3
        push!(cells, QuadraticQuadrilateral((node_array3[2i-1,2j], 
                                             node_array3[2i+1,2j], 
                                             node_array3[2i+1,2j+2], 
                                             node_array3[2i-1,2j+2],
                                             node_array3[2i,2j], 
                                             node_array3[2i+1,2j+1], 
                                             node_array3[2i,2j+2], 
                                             node_array3[2i-1,2j+1], 
                                             node_array3[2i,2j+1])))

        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
        if j == nel_y3
            push!(boundary, (length(cells), 3))
            push!(facesets["top"], (length(cells), 3))
        end
    end
    
    boundary_matrix = Ferrite.boundaries_to_sparse(boundary)

    return Grid(cells, nodes, facesets=facesets, nodesets=nodesets, 
        boundary_matrix=boundary_matrix)
end

function TieBeamGrid(::Type{Val{CellType}}, ::Type{T}=Float64, refine=1) where {T, CellType}
    if CellType === :Linear
        return _LinearTieBeamGrid(T, refine)
    else
        return _QuadraticTieBeamGrid(T, refine)
    end
end

function _LinearTieBeamGrid(::Type{T}=Float64, refine=1) where {T}
    nodes = Node{2,T}[]
    cells = Quadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["leftfixed"]  = Set{Tuple{Int,Int}}()
    facesets["toproller"]    = Set{Tuple{Int,Int}}()
    facesets["rightload"]  = Set{Tuple{Int,Int}}()
    facesets["bottomload"] = Set{Tuple{Int,Int}}()
    
    # Lower left rectangle
    nel_x1 = 32 * refine; nel_y1 = 3 * refine;
    n_nodes_x1 = nel_x1 + 1; n_nodes_y1 = nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1
    
    LL = Vec{2,T}((0, 0))
    LR = Vec{2,T}((T(nel_x1 / refine), T(0)))
    UR = Vec{2,T}((T(nel_x1 / refine), T(nel_y1 / refine)))
    UL = Vec{2,T}((T(0), T(nel_y1 / refine)))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, LR, UR, UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(cells, Quadrilateral((node_array1[i,j], node_array1[i+1,j], 
                                    node_array1[i+1,j+1],node_array1[i,j+1])))
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
            push!(facesets["leftfixed"], (cidx, 4))
        end
        if i == nel_x1
            cidx = length(cells)
            push!(boundary, (cidx, 2))
            push!(facesets["rightload"], (cidx, 2))
        end
        if j == 1
            cidx = length(cells)
            push!(boundary, (cidx, 1))
            if i == 31
                push!(facesets["bottomload"], (cidx, 1))
            end
        end
        if j == nel_y1 && i != 31
            cidx = length(cells)
            push!(boundary, (cidx, 3))
        end
    end

    nel_x2 = 1 * refine; nel_y2 = 3 * refine + refine - 1
    n_nodes_x2 = nel_x2 + 1; n_nodes_y2 = nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2
    indexoffset = length(nodes)
    LL = Vec{2,T}((T(30), (nel_y1 + T(1)) / refine))
    LR = Vec{2,T}((T(31), (nel_y1 + T(1)) / refine))
    UR = Vec{2,T}((T(31), nel_y1 / refine + T(4)))
    UL = Vec{2,T}((T(30), nel_y1 / refine + T(4)))
    
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, LL, LR, UR, UL)
    node_array2 = reshape(collect(indexoffset+1:indexoffset+n_nodes2), (n_nodes_x2, n_nodes_y2))

    t = 30
    for i in 1:refine
        push!(cells, Quadrilateral((node_array1[t*refine+i, nel_y1 + 1], 
                                    node_array1[t*refine+i+1, nel_y1 + 1], 
                                    node_array2[i+1,1], 
                                    node_array2[i,1])))
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
        end
        if i == refine
            cidx = length(cells)
            push!(boundary, (cidx, 2))
        end
    end

    for j in 1:nel_y2, i in 1:nel_x2
        push!(cells, Quadrilateral((node_array2[i,j], node_array2[i+1,j], 
                                    node_array2[i+1,j+1], node_array2[i,j+1])))
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
        end
        if i == nel_x2
            cidx = length(cells)
            push!(boundary, (cidx, 2))
        end
        if j == nel_y2
            cidx = length(cells)
            push!(boundary, (cidx, 3))
            push!(facesets["toproller"], (cidx, 3))
        end
    end

    boundary_matrix = Ferrite.boundaries_to_sparse(boundary)
    return Grid(cells, nodes, facesets=facesets, 
        boundary_matrix=boundary_matrix)
end

function _QuadraticTieBeamGrid(::Type{T}=Float64, refine = 1) where {T}
    nodes = Node{2,T}[]
    cells = QuadraticQuadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["leftfixed"]  = Set{Tuple{Int,Int}}()
    facesets["toproller"]    = Set{Tuple{Int,Int}}()
    facesets["rightload"]  = Set{Tuple{Int,Int}}()
    facesets["bottomload"] = Set{Tuple{Int,Int}}()
    
    # Lower left rectangle
    nel_x1 = 32*refine; nel_y1 = 3*refine;
    n_nodes_x1 = 2*nel_x1 + 1; n_nodes_y1 = 2*nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1
    
    LL = Vec{2,T}((0, 0))
    LR = Vec{2,T}((T(nel_x1/refine), T(0)))
    UR = Vec{2,T}((T(nel_x1/refine), T(nel_y1/refine)))
    UL = Vec{2,T}((T(0), T(nel_y1/refine)))
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, LR, UR, UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(cells, QuadraticQuadrilateral((node_array1[2*i-1,2*j-1], node_array1[2*i+1,2*j-1], 
                                             node_array1[2*i+1,2*j+1],node_array1[2*i-1,2*j+1], 
                                             node_array1[2*i,2*j-1],node_array1[2*i+1,2*j], 
                                             node_array1[2*i,2*j+1],node_array1[2*i-1,2*j], 
                                             node_array1[2*i,2*j])))
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
            push!(facesets["leftfixed"], (cidx, 4))
        end
        if i == nel_x1
            cidx = length(cells)
            push!(boundary, (cidx, 2))
            push!(facesets["rightload"], (cidx, 2))
        end
        if j == 1
            cidx = length(cells)
            push!(boundary, (cidx, 1))
            if i == 31
                push!(facesets["bottomload"], (cidx, 1))
            end
        end
        if j == nel_y1 && i != 31
            cidx = length(cells)
            push!(boundary, (cidx, 3))
        end
    end

    nel_x2 = 1*refine; nel_y2 = 3*refine + refine - 1
    n_nodes_x2 = 2*nel_x2 + 1; n_nodes_y2 = 2*nel_y2 + 2
    n_nodes2 = n_nodes_x2 * n_nodes_y2
    indexoffset = length(nodes)
    LL = Vec{2,T}((T(30), (nel_y1 + T(0.5)) / refine))
    LR = Vec{2,T}((T(31), (nel_y1 + T(0.5)) / refine))
    UR = Vec{2,T}((T(31), nel_y1/refine + T(4)))
    UL = Vec{2,T}((T(30), nel_y1/refine + T(4)))
    
    Ferrite._generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, LL, LR, UR, UL)
    node_array2 = reshape(collect(indexoffset+1:indexoffset+n_nodes2), (n_nodes_x2, n_nodes_y2))

    t = 30
    for i in 1:refine
        push!(cells, QuadraticQuadrilateral((node_array1[2*(refine*t+i-1)+1, 2*nel_y1+1], 
                                            node_array1[2*(refine*t+i-1)+3, 2*nel_y1+1], 
                                            node_array2[1+2i, 2], 
                                            node_array2[2i-1, 2], 
                                            node_array1[2*(refine*t+i-1)+2, 2*nel_y1+1], 
                                            node_array2[1+2i, 1], 
                                            node_array2[2i, 2], 
                                            node_array2[2i-1, 1],
                                            node_array2[2i, 1])))

        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
        end
        if i == refine
            cidx = length(cells)
            push!(boundary, (cidx, 2))        
        end
    end

    for j in 1:nel_y2, i in 1:nel_x2
        push!(cells, QuadraticQuadrilateral((node_array2[2*i-1,2*j], node_array2[2*i+1,2*j], 
                                             node_array2[2*i+1,2*j+2],node_array2[2*i-1,2*j+2], 
                                             node_array2[2*i,2*j],node_array2[2*i+1,2*j+1], 
                                             node_array2[2*i,2*j+2],node_array2[2*i-1,2*j+1], 
                                             node_array2[2*i,2*j+1])))
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
        end
        if i == nel_x2
            cidx = length(cells)
            push!(boundary, (cidx, 2))
        end
        if j == nel_y2
            cidx = length(cells)
            push!(boundary, (cidx, 3))
            push!(facesets["toproller"], (cidx, 3))
        end
    end

    boundary_matrix = Ferrite.boundaries_to_sparse(boundary)
    return Grid(cells, nodes, facesets=facesets, 
        boundary_matrix=boundary_matrix)
end
