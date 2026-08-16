abstract type AbstractGrid{dim,T} end

const Vec = Ferrite.Vec

"""
```
struct RectilinearGrid{dim, T, N, M, TG<:Ferrite.Grid{dim, <:Ferrite.Cell{dim,N,M}, T}} <: AbstractGrid{dim, T}
    grid::TG
    nels::NTuple{dim, Int}
    sizes::NTuple{dim, T}
    corners::NTuple{2, Vec{dim, T}}
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
"""
struct RectilinearGrid{dim,T,N,M,TG<:Ferrite.AbstractGrid{dim}} <: AbstractGrid{dim,T}
    grid::TG
    nels::NTuple{dim,Int}
    sizes::NTuple{dim,T}
    corners::NTuple{2,Vec{dim,T}}
end

"""
    _celltype_tag(celltype::Symbol)

Return the `Val{:Linear}` or `Val{:Quadratic}` tag corresponding to the
user-facing `celltype` keyword. Throws an `ArgumentError` for any other value.
Used as a function barrier so the cell type is a compile-time constant inside
the internal constructors.
"""
function _celltype_tag(celltype::Symbol)
    celltype === :Linear && return Val(:Linear)
    celltype === :Quadratic && return Val(:Quadratic)
    throw(ArgumentError("celltype must be :Linear or :Quadratic, got :$celltype"))
end

"""
    RectilinearGrid(nels::NTuple{dim,Int}, sizes::NTuple{dim,T}; celltype=:Linear) where {dim, T}

Constructs an instance of [`RectilinearGrid`](@ref).

- `dim`: dimension of the problem
- `T`: number type for coordinates
- `nels`: number of elements in every dimension
- `sizes`: dimensions of each rectilinear cell
- `celltype`: either `:Linear` or `:Quadratic` to determine the order of the
  geometric and field basis functions and element type. Only isoparametric
  elements are supported for now.

Example:

```
rectgrid = RectilinearGrid((60,20), (1.0,1.0))
```
"""
function RectilinearGrid(
    nels::NTuple{<:Any,Int}, sizes::NTuple{<:Any}; celltype::Symbol=:Linear
)
    return _RectilinearGrid(_celltype_tag(celltype), nels, sizes)
end

function _RectilinearGrid(::Val{:Linear}, nels::NTuple{<:Any,Int}, sizes::NTuple{<:Any})
    return _RectilinearGrid_Linear(Val(length(nels)), eltype(sizes), nels, sizes)
end

function _RectilinearGrid(::Val{:Quadratic}, nels::NTuple{<:Any,Int}, sizes::NTuple{<:Any})
    return _RectilinearGrid_Quadratic(Val(length(nels)), eltype(sizes), nels, sizes)
end

function _RectilinearGrid_Linear(
    ::Val{dim}, ::Type{T}, nels::NTuple{dim,Int}, sizes::NTuple{dim,T}
) where {dim,T}
    geoshape = dim === 2 ? Quadrilateral : Hexahedron
    corner1 = Vec{dim}(fill(T(0), dim))
    corner2 = Vec{dim}((nels .* sizes))
    grid = generate_grid(geoshape, nels, corner1, corner2)

    N = nnodes(geoshape)
    M = Ferrite.nfacets(Ferrite.getrefshape(geoshape))
    ncells = prod(nels)
    return RectilinearGrid{dim,T,N,M,typeof(grid)}(grid, nels, sizes, (corner1, corner2))
end

function _RectilinearGrid_Quadratic(
    ::Val{dim}, ::Type{T}, nels::NTuple{dim,Int}, sizes::NTuple{dim,T}
) where {dim,T}
    geoshape = dim === 2 ? QuadraticQuadrilateral : Hexahedron
    corner1 = Vec{dim}(fill(T(0), dim))
    corner2 = Vec{dim}((nels .* sizes))
    grid = generate_grid(geoshape, nels, corner1, corner2)

    N = nnodes(geoshape)
    M = Ferrite.nfacets(Ferrite.getrefshape(geoshape))
    ncells = prod(nels)
    return RectilinearGrid{dim,T,N,M,typeof(grid)}(grid, nels, sizes, (corner1, corner2))
end

nnodespercell(::RectilinearGrid{dim,T,N,M}) where {dim,T,N,M} = N
nfacespercell(::RectilinearGrid{dim,T,N,M}) where {dim,T,N,M} = M

left(rectgrid::RectilinearGrid, x) = x[1] ≈ rectgrid.corners[1][1]
right(rectgrid::RectilinearGrid, x) = x[1] ≈ rectgrid.corners[2][1]
bottom(rectgrid::RectilinearGrid, x) = x[2] ≈ rectgrid.corners[1][2]
top(rectgrid::RectilinearGrid, x) = x[2] ≈ rectgrid.corners[2][2]
back(rectgrid::RectilinearGrid, x) = x[3] ≈ rectgrid.corners[1][3]
front(rectgrid::RectilinearGrid, x) = x[3] ≈ rectgrid.corners[2][3]
function middlex(rectgrid::RectilinearGrid, x)
    return x[1] ≈ (rectgrid.corners[1][1] + rectgrid.corners[2][1]) / 2
end
function middley(rectgrid::RectilinearGrid, x)
    return x[2] ≈ (rectgrid.corners[1][2] + rectgrid.corners[2][2]) / 2
end
function middlez(rectgrid::RectilinearGrid, x)
    return x[3] ≈ (rectgrid.corners[1][3] + rectgrid.corners[2][3]) / 2
end

nnodes(cell::Type{<:Ferrite.AbstractCell}) = length(Base.fieldtypes(cell)[1].parameters)
nnodes(cell::Ferrite.AbstractCell) = length(cell.nodes)

"""
    LGrid(::Type{T}; celltype=:Linear, length = 100, height = 100, upperslab = 50, lowerslab = 50) where {T}
    LGrid(nel1::NTuple{2,Int}, nel2::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T}, MR::Vec{2,T}; celltype=:Linear) where {T}

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

`celltype` is either `:Linear` or `:Quadratic` to determine the order of the
geometric and field basis functions and element type. Only isoparametric
elements are supported for now.

Examples:

```
LGrid(upperslab = 30, lowerslab = 70)
LGrid((2, 4), (2, 2), Vec{2,Float64}((0.0,0.0)), Vec{2,Float64}((2.0, 4.0)), Vec{2,Float64}((4.0, 2.0)))
```
"""
function LGrid(
    ::Type{T};
    celltype::Symbol=:Linear,
    length=100,
    height=100,
    upperslab=50,
    lowerslab=50,
    load_width=nothing,
) where {T}
    length > upperslab || throw(
        ArgumentError(
            "LGrid: length ($length) must be greater than upperslab ($upperslab)"
        ),
    )
    height > lowerslab || throw(
        ArgumentError(
            "LGrid: height ($height) must be greater than lowerslab ($lowerslab)"
        ),
    )
    return LGrid(
        (upperslab, height),
        (length - upperslab, lowerslab),
        Vec{2,T}((0.0, 0.0)),
        Vec{2,T}((T(upperslab), T(height))),
        Vec{2,T}((T(length), T(lowerslab)));
        celltype=celltype,
        load_width=load_width,
    )
end
function LGrid(
    nel1::NTuple{2,Int},
    nel2::NTuple{2,Int},
    LL::Vec{2,T},
    UR::Vec{2,T},
    MR::Vec{2,T};
    celltype::Symbol=:Linear,
    load_width=nothing,
) where {T}
    return _LGrid(_celltype_tag(celltype), nel1, nel2, LL, UR, MR; load_width=load_width)
end

function _LGrid(
    ::Val{:Linear},
    nel1::NTuple{2,Int},
    nel2::NTuple{2,Int},
    LL::Vec{2,T},
    UR::Vec{2,T},
    MR::Vec{2,T};
    load_width=nothing,
) where {T}
    return _LinearLGrid(nel1, nel2, LL, UR, MR; load_width=load_width)
end

function _LGrid(
    ::Val{:Quadratic},
    nel1::NTuple{2,Int},
    nel2::NTuple{2,Int},
    LL::Vec{2,T},
    UR::Vec{2,T},
    MR::Vec{2,T};
    load_width=nothing,
) where {T}
    return _QuadraticLGrid(nel1, nel2, LL, UR, MR; load_width=load_width)
end

"""
    _load_nodes!(nodeset, node_array, midpointindy, load_width)

Populate the "load" node set of an L-shaped grid: a single midpoint node of
the right edge when `load_width === nothing`, otherwise `load_width`
consecutive nodes centered on the midpoint (clamped to the edge).
"""
function _load_nodes!(nodeset, node_array, midpointindy, load_width)
    if load_width === nothing
        push!(nodeset, node_array[end, midpointindy])
        return nodeset
    end
    load_width isa Integer && load_width >= 1 ||
        throw(ArgumentError("load_width must be a positive integer, got $load_width"))
    first = max(1, midpointindy - load_width ÷ 2)
    last = min(size(node_array, 2), first + load_width - 1)
    for j in first:last
        push!(nodeset, node_array[end, j])
    end
    return nodeset
end

function _generate_2d_nodes!(nodes, nx, ny, LL, LR, UR, UL)
    for j in 1:ny, i in 1:nx
        s = (i - 1) / (nx - 1)
        t = (j - 1) / (ny - 1)
        x =
            (1 - s) * (1 - t) * LL[1] +
            s * (1 - t) * LR[1] +
            s * t * UR[1] +
            (1 - s) * t * UL[1]
        y =
            (1 - s) * (1 - t) * LL[2] +
            s * (1 - t) * LR[2] +
            s * t * UR[2] +
            (1 - s) * t * UL[2]
        push!(nodes, Node(Vec{2,typeof(x)}((x, y))))
    end
    return nodes
end

function _LinearLGrid(
    nel1::NTuple{2,Int},
    nel2::NTuple{2,Int},
    LL::Vec{2,T},
    UR::Vec{2,T},
    MR::Vec{2,T};
    load_width=nothing,
) where {T}
    nel1[2] > nel2[2] || throw(
        ArgumentError(
            "_LinearLGrid: nel1[2] ($(nel1[2])) must be greater than nel2[2] ($(nel2[2]))",
        ),
    )

    midpointindy = round(Int, nel2[2] / 2) + 1
    nodes = Node{2,T}[]
    cells = Quadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String,Set{Ferrite.FacetIndex}}()
    facesets["right"] = Set{Ferrite.FacetIndex}()
    facesets["top"] = Set{Ferrite.FacetIndex}()
    nodesets = Dict{String,Set{Int}}()
    nodesets["load"] = Set{Int}()

    # Lower left rectangle
    nel_x1 = nel1[1]
    nel_y1 = nel2[2]
    n_nodes_x1 = nel_x1 + 1
    n_nodes_y1 = nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    _LR = Vec{2,T}((UR[1], LL[2]))
    _UL = Vec{2,T}((LL[1], MR[2]))
    _UR = Vec{2,T}((UR[1], MR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, _LR, _UR, _UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(
            cells,
            Quadrilateral((
                node_array1[i, j],
                node_array1[i + 1, j],
                node_array1[i + 1, j + 1],
                node_array1[i, j + 1],
            )),
        )
        if i == 1
            push!(boundary, (length(cells), 4))
        end
        if j == 1
            push!(boundary, (length(cells), 1))
        end
    end

    # Lower right rectangle
    offsetstep = (MR[1] - _LR[1]) / nel2[1]
    indexoffset = length(nodes)

    nel_x2 = nel2[1] - 1
    nel_y2 = nel2[2]
    n_nodes_x2 = nel_x2 + 1
    n_nodes_y2 = nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2

    _LL = Vec{2,T}((_LR[1] + offsetstep, _LR[2]))
    _LR = Vec{2,T}((MR[1], LL[2]))
    _UL = Vec{2,T}((_UR[1] + offsetstep, MR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, _LL, _LR, MR, _UL)

    node_array2 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes2)), (n_nodes_x2, n_nodes_y2)
    )
    for j in 1:nel_y2
        push!(
            cells,
            Quadrilateral((
                node_array1[end, j],
                node_array2[1, j],
                node_array2[1, j + 1],
                node_array1[end, j + 1],
            )),
        )
        j == 1 && push!(boundary, (length(cells), 1))
        j == nel_y2 && push!(boundary, (length(cells), 3))
        if nel_x2 == 1
            push!(boundary, (length(cells), 2))
            push!(facesets["right"], Ferrite.FacetIndex(length(cells), 2))
        end
        for i in 1:nel_x2
            push!(
                cells,
                Quadrilateral((
                    node_array2[i, j],
                    node_array2[i + 1, j],
                    node_array2[i + 1, j + 1],
                    node_array2[i, j + 1],
                )),
            )
            if i == nel_x2
                push!(boundary, (length(cells), 2))
                push!(facesets["right"], Ferrite.FacetIndex(length(cells), 2))
            end
            j == 1 && push!(boundary, (length(cells), 1))
            j == nel_y2 && push!(boundary, (length(cells), 3))
        end
    end

    _load_nodes!(nodesets["load"], node_array2, midpointindy, load_width)

    # Upper left rectangle
    offsetstep = (UR[2] - MR[2]) / (nel1[2] - nel2[2])
    indexoffset = length(nodes)

    nel_x3 = nel1[1]
    nel_y3 = nel1[2] - nel2[2] - 1
    n_nodes_x3 = nel_x3 + 1
    n_nodes_y3 = nel_y3 + 1
    n_nodes3 = n_nodes_x3 * n_nodes_y3

    _LL = Vec{2,T}((LL[1], MR[2] + offsetstep))
    _LR = Vec{2,T}((UR[1], MR[2] + offsetstep))
    _UL = Vec{2,T}((LL[1], UR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x3, n_nodes_y3, _LL, _LR, UR, _UL)

    # Generate cells
    node_array3 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes3)), (n_nodes_x3, n_nodes_y3)
    )

    for i in 1:nel_x3
        push!(
            cells,
            Quadrilateral((
                node_array1[i, end],
                node_array1[i + 1, end],
                node_array3[i + 1, 1],
                node_array3[i, 1],
            )),
        )
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
    end
    for j in 1:nel_y3, i in 1:nel_x3
        push!(
            cells,
            Quadrilateral((
                node_array3[i, j],
                node_array3[i + 1, j],
                node_array3[i + 1, j + 1],
                node_array3[i, j + 1],
            )),
        )
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
        if j == nel_y3
            push!(boundary, (length(cells), 3))
            push!(facesets["top"], Ferrite.FacetIndex(length(cells), 3))
        end
    end

    facesets["boundary"] = Set{Ferrite.FacetIndex}(
        Ferrite.FacetIndex(c, f) for (c, f) in boundary
    )
    return Grid(cells, nodes; facetsets=facesets, nodesets=nodesets)
end

function _QuadraticLGrid(
    nel1::NTuple{2,Int},
    nel2::NTuple{2,Int},
    LL::Vec{2,T},
    UR::Vec{2,T},
    MR::Vec{2,T};
    load_width=nothing,
) where {T}
    nel1[2] > nel2[2] || throw(
        ArgumentError(
            "_QuadraticLGrid: nel1[2] ($(nel1[2])) must be greater than nel2[2] ($(nel2[2]))",
        ),
    )

    midpointindy = round(Int, nel2[2] / 2) + 1
    nodes = Node{2,T}[]
    cells = QuadraticQuadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String,Set{Ferrite.FacetIndex}}()
    facesets["right"] = Set{Ferrite.FacetIndex}()
    facesets["top"] = Set{Ferrite.FacetIndex}()
    nodesets = Dict{String,Set{Int}}()
    nodesets["load"] = Set{Int}()

    # Lower left rectangle
    nel_x1 = nel1[1]
    nel_y1 = nel2[2]
    n_nodes_x1 = 2 * nel_x1 + 1
    n_nodes_y1 = 2 * nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    _LR = Vec{2,T}((UR[1], LL[2]))
    _UL = Vec{2,T}((LL[1], MR[2]))
    _UR = Vec{2,T}((UR[1], MR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, _LR, _UR, _UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array1[2 * i - 1, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j + 1],
                node_array1[2 * i - 1, 2 * j + 1],
                node_array1[2 * i, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j],
                node_array1[2 * i, 2 * j + 1],
                node_array1[2 * i - 1, 2 * j],
                node_array1[2 * i, 2 * j],
            )),
        )
        if i == 1
            push!(boundary, (length(cells), 4))
        end
        if j == 1
            push!(boundary, (length(cells), 1))
        end
    end

    # Lower right rectangle
    offsetstep = (MR[1] - _LR[1]) / nel2[1] / 2
    indexoffset = length(nodes)

    nel_x2 = nel2[1] - 1
    nel_y2 = nel2[2]
    n_nodes_x2 = 2 * nel_x2 + 2
    n_nodes_y2 = 2 * nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2

    _LL = Vec{2,T}((_LR[1] + offsetstep, _LR[2]))
    _LR = Vec{2,T}((MR[1], LL[2]))
    _UL = Vec{2,T}((_UR[1] + offsetstep, MR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, _LL, _LR, MR, _UL)

    node_array2 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes2)), (n_nodes_x2, n_nodes_y2)
    )
    for j in 1:nel_y2
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array1[end, 2 * j - 1],
                node_array2[2, 2 * j - 1],
                node_array2[2, 2 * j + 1],
                node_array1[end, 2 * j + 1],
                node_array2[1, 2 * j - 1],
                node_array2[2, 2 * j],
                node_array2[1, 2 * j + 1],
                node_array1[end, 2 * j],
                node_array2[1, 2 * j],
            )),
        )
        j == 1 && push!(boundary, (length(cells), 1))
        j == nel_y2 && push!(boundary, (length(cells), 3))
        if nel_x2 == 1
            push!(boundary, (length(cells), 2))
            push!(facesets["right"], Ferrite.FacetIndex(length(cells), 2))
        end
        for i in 1:nel_x2
            push!(
                cells,
                QuadraticQuadrilateral((
                    node_array2[2 * i, 2 * j - 1],
                    node_array2[2 * i + 2, 2 * j - 1],
                    node_array2[2 * i + 2, 2 * j + 1],
                    node_array2[2 * i, 2 * j + 1],
                    node_array2[2 * i + 1, 2 * j - 1],
                    node_array2[2 * i + 2, 2 * j],
                    node_array2[2 * i + 1, 2 * j + 1],
                    node_array2[2 * i, 2 * j],
                    node_array2[2 * i + 1, 2 * j],
                )),
            )
            if i == nel_x2
                push!(boundary, (length(cells), 2))
                push!(facesets["right"], Ferrite.FacetIndex(length(cells), 2))
            end
            j == 1 && push!(boundary, (length(cells), 1))
            j == nel_y2 && push!(boundary, (length(cells), 3))
        end
    end

    _load_nodes!(nodesets["load"], node_array2, midpointindy, load_width)

    # Upper left rectangle
    offsetstep = (UR[2] - MR[2]) / (nel1[2] - nel2[2]) / 2
    indexoffset = length(nodes)

    nel_x3 = nel1[1]
    nel_y3 = nel1[2] - nel2[2] - 1
    n_nodes_x3 = 2 * nel_x3 + 1
    n_nodes_y3 = 2 * nel_y3 + 2
    n_nodes3 = n_nodes_x3 * n_nodes_y3

    _LL = Vec{2,T}((LL[1], MR[2] + offsetstep))
    _LR = Vec{2,T}((UR[1], MR[2] + offsetstep))
    _UL = Vec{2,T}((LL[1], UR[2]))
    _generate_2d_nodes!(nodes, n_nodes_x3, n_nodes_y3, _LL, _LR, UR, _UL)

    # Generate cells
    node_array3 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes3)), (n_nodes_x3, n_nodes_y3)
    )

    for i in 1:nel_x3
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array1[2i - 1, end],
                node_array1[2i + 1, end],
                node_array3[2i + 1, 2],
                node_array3[2i - 1, 2],
                node_array1[2i, end],
                node_array3[2i + 1, 1],
                node_array3[2i, 2],
                node_array3[2i - 1, 1],
                node_array3[2i, 1],
            )),
        )
        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
    end
    for j in 1:nel_y3, i in 1:nel_x3
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array3[2i - 1, 2j],
                node_array3[2i + 1, 2j],
                node_array3[2i + 1, 2j + 2],
                node_array3[2i - 1, 2j + 2],
                node_array3[2i, 2j],
                node_array3[2i + 1, 2j + 1],
                node_array3[2i, 2j + 2],
                node_array3[2i - 1, 2j + 1],
                node_array3[2i, 2j + 1],
            )),
        )

        i == 1 && push!(boundary, (length(cells), 4))
        i == nel_x3 && push!(boundary, (length(cells), 2))
        if j == nel_y3
            push!(boundary, (length(cells), 3))
            push!(facesets["top"], Ferrite.FacetIndex(length(cells), 3))
        end
    end

    facesets["boundary"] = Set{Ferrite.FacetIndex}(
        Ferrite.FacetIndex(c, f) for (c, f) in boundary
    )
    return Grid(cells, nodes; facetsets=facesets, nodesets=nodesets)
end

function TieBeamGrid((::Type{T})=Float64; celltype::Symbol=:Linear, refine=1) where {T}
    return _TieBeamGrid(_celltype_tag(celltype), T, refine)
end

function _TieBeamGrid(::Val{:Linear}, ::Type{T}, refine=1) where {T}
    return _LinearTieBeamGrid(T, refine)
end

function _TieBeamGrid(::Val{:Quadratic}, ::Type{T}, refine=1) where {T}
    return _QuadraticTieBeamGrid(T, refine)
end

function _LinearTieBeamGrid((::Type{T})=Float64, refine=1) where {T}
    nodes = Node{2,T}[]
    cells = Quadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String,Set{Ferrite.FacetIndex}}()
    facesets["leftfixed"] = Set{Ferrite.FacetIndex}()
    facesets["toproller"] = Set{Ferrite.FacetIndex}()
    facesets["rightload"] = Set{Ferrite.FacetIndex}()
    facesets["bottomload"] = Set{Ferrite.FacetIndex}()

    # Lower left rectangle
    nel_x1 = 32 * refine
    nel_y1 = 3 * refine
    n_nodes_x1 = nel_x1 + 1
    n_nodes_y1 = nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    LL = Vec{2,T}((0, 0))
    LR = Vec{2,T}((T(nel_x1 / refine), T(0)))
    UR = Vec{2,T}((T(nel_x1 / refine), T(nel_y1 / refine)))
    UL = Vec{2,T}((T(0), T(nel_y1 / refine)))
    _generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, LR, UR, UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(
            cells,
            Quadrilateral((
                node_array1[i, j],
                node_array1[i + 1, j],
                node_array1[i + 1, j + 1],
                node_array1[i, j + 1],
            )),
        )
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
            push!(facesets["leftfixed"], Ferrite.FacetIndex(cidx, 4))
        end
        if i == nel_x1
            cidx = length(cells)
            push!(boundary, (cidx, 2))
            push!(facesets["rightload"], Ferrite.FacetIndex(cidx, 2))
        end
        if j == 1
            cidx = length(cells)
            push!(boundary, (cidx, 1))
            if i == 31
                push!(facesets["bottomload"], Ferrite.FacetIndex(cidx, 1))
            end
        end
        if j == nel_y1 && i != 31
            cidx = length(cells)
            push!(boundary, (cidx, 3))
        end
    end

    nel_x2 = 1 * refine
    nel_y2 = 3 * refine + refine - 1
    n_nodes_x2 = nel_x2 + 1
    n_nodes_y2 = nel_y2 + 1
    n_nodes2 = n_nodes_x2 * n_nodes_y2
    indexoffset = length(nodes)
    LL = Vec{2,T}((T(30), (nel_y1 + T(1)) / refine))
    LR = Vec{2,T}((T(31), (nel_y1 + T(1)) / refine))
    UR = Vec{2,T}((T(31), nel_y1 / refine + T(4)))
    UL = Vec{2,T}((T(30), nel_y1 / refine + T(4)))

    _generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, LL, LR, UR, UL)
    node_array2 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes2)), (n_nodes_x2, n_nodes_y2)
    )

    t = 30
    for i in 1:refine
        push!(
            cells,
            Quadrilateral((
                node_array1[t * refine + i, nel_y1 + 1],
                node_array1[t * refine + i + 1, nel_y1 + 1],
                node_array2[i + 1, 1],
                node_array2[i, 1],
            )),
        )
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
        push!(
            cells,
            Quadrilateral((
                node_array2[i, j],
                node_array2[i + 1, j],
                node_array2[i + 1, j + 1],
                node_array2[i, j + 1],
            )),
        )
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
            push!(facesets["toproller"], Ferrite.FacetIndex(cidx, 3))
        end
    end

    facesets["boundary"] = Set{Ferrite.FacetIndex}(
        Ferrite.FacetIndex(c, f) for (c, f) in boundary
    )
    return Grid(cells, nodes; facetsets=facesets)
end

function _QuadraticTieBeamGrid((::Type{T})=Float64, refine=1) where {T}
    nodes = Node{2,T}[]
    cells = QuadraticQuadrilateral[]
    boundary = Tuple{Int,Int}[]
    facesets = Dict{String,Set{Ferrite.FacetIndex}}()
    facesets["leftfixed"] = Set{Ferrite.FacetIndex}()
    facesets["toproller"] = Set{Ferrite.FacetIndex}()
    facesets["rightload"] = Set{Ferrite.FacetIndex}()
    facesets["bottomload"] = Set{Ferrite.FacetIndex}()

    # Lower left rectangle
    nel_x1 = 32 * refine
    nel_y1 = 3 * refine
    n_nodes_x1 = 2 * nel_x1 + 1
    n_nodes_y1 = 2 * nel_y1 + 1
    n_nodes1 = n_nodes_x1 * n_nodes_y1

    LL = Vec{2,T}((0, 0))
    LR = Vec{2,T}((T(nel_x1 / refine), T(0)))
    UR = Vec{2,T}((T(nel_x1 / refine), T(nel_y1 / refine)))
    UL = Vec{2,T}((T(0), T(nel_y1 / refine)))
    _generate_2d_nodes!(nodes, n_nodes_x1, n_nodes_y1, LL, LR, UR, UL)

    node_array1 = reshape(collect(1:n_nodes1), (n_nodes_x1, n_nodes_y1))
    for j in 1:nel_y1, i in 1:nel_x1
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array1[2 * i - 1, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j + 1],
                node_array1[2 * i - 1, 2 * j + 1],
                node_array1[2 * i, 2 * j - 1],
                node_array1[2 * i + 1, 2 * j],
                node_array1[2 * i, 2 * j + 1],
                node_array1[2 * i - 1, 2 * j],
                node_array1[2 * i, 2 * j],
            )),
        )
        if i == 1
            cidx = length(cells)
            push!(boundary, (cidx, 4))
            push!(facesets["leftfixed"], Ferrite.FacetIndex(cidx, 4))
        end
        if i == nel_x1
            cidx = length(cells)
            push!(boundary, (cidx, 2))
            push!(facesets["rightload"], Ferrite.FacetIndex(cidx, 2))
        end
        if j == 1
            cidx = length(cells)
            push!(boundary, (cidx, 1))
            if i == 31
                push!(facesets["bottomload"], Ferrite.FacetIndex(cidx, 1))
            end
        end
        if j == nel_y1 && i != 31
            cidx = length(cells)
            push!(boundary, (cidx, 3))
        end
    end

    nel_x2 = 1 * refine
    nel_y2 = 3 * refine + refine - 1
    n_nodes_x2 = 2 * nel_x2 + 1
    n_nodes_y2 = 2 * nel_y2 + 2
    n_nodes2 = n_nodes_x2 * n_nodes_y2
    indexoffset = length(nodes)
    LL = Vec{2,T}((T(30), (nel_y1 + T(0.5)) / refine))
    LR = Vec{2,T}((T(31), (nel_y1 + T(0.5)) / refine))
    UR = Vec{2,T}((T(31), nel_y1 / refine + T(4)))
    UL = Vec{2,T}((T(30), nel_y1 / refine + T(4)))

    _generate_2d_nodes!(nodes, n_nodes_x2, n_nodes_y2, LL, LR, UR, UL)
    node_array2 = reshape(
        collect((indexoffset + 1):(indexoffset + n_nodes2)), (n_nodes_x2, n_nodes_y2)
    )

    t = 30
    for i in 1:refine
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array1[2 * (refine * t + i - 1) + 1, 2 * nel_y1 + 1],
                node_array1[2 * (refine * t + i - 1) + 3, 2 * nel_y1 + 1],
                node_array2[1 + 2i, 2],
                node_array2[2i - 1, 2],
                node_array1[2 * (refine * t + i - 1) + 2, 2 * nel_y1 + 1],
                node_array2[1 + 2i, 1],
                node_array2[2i, 2],
                node_array2[2i - 1, 1],
                node_array2[2i, 1],
            )),
        )

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
        push!(
            cells,
            QuadraticQuadrilateral((
                node_array2[2 * i - 1, 2 * j],
                node_array2[2 * i + 1, 2 * j],
                node_array2[2 * i + 1, 2 * j + 2],
                node_array2[2 * i - 1, 2 * j + 2],
                node_array2[2 * i, 2 * j],
                node_array2[2 * i + 1, 2 * j + 1],
                node_array2[2 * i, 2 * j + 2],
                node_array2[2 * i - 1, 2 * j + 1],
                node_array2[2 * i, 2 * j + 1],
            )),
        )
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
            push!(facesets["toproller"], Ferrite.FacetIndex(cidx, 3))
        end
    end

    facesets["boundary"] = Set{Ferrite.FacetIndex}(
        Ferrite.FacetIndex(c, f) for (c, f) in boundary
    )
    return Grid(cells, nodes; facetsets=facesets)
end
