using ..TopOpt.TopOptProblems:
    StiffnessTopOptProblem, Metadata, RectilinearGrid, left, right, middley, middlez

get_fixities_node_set_name(i) = "fixed_u$(i)"

struct TrussProblem{
    xdim,
    T,
    N,
    M,
    Tt<:TrussGrid{xdim,T,N,M},
    Tm1<:Vector{<:TrussFEAMaterial{T}},
    Tc<:ConstraintHandler{<:DofHandler{xdim,<:Ferrite.Cell{xdim,N,M},T},T},
    Tb<:AbstractVector,
    Tw<:AbstractVector,
    Tv<:AbstractVector{Int},
    Tm2<:Metadata,
} <: StiffnessTopOptProblem{xdim,T}
    truss_grid::Tt # ground truss mesh
    materials::Tm1
    ch::Tc
    force::Dict{Int,SVector{xdim,T}}
    black::Tb
    white::Tw
    varind::Tv # variable dof => free dof, based on black & white
    metadata::Tm2
end
# - `force_dof`: dof number at which the force is applied

gettrussgrid(sp::TrussProblem) = sp.truss_grid
getFerritegrid(sp::TrussProblem) = sp.truss_grid.grid

TopOpt.TopOptProblems.getE(sp::TrussProblem) = [m.E for m in sp.materials]
TopOpt.TopOptProblems.getν(sp::TrussProblem) = [m.ν for m in sp.materials]
getA(sp::TrussProblem) = [cs.A for cs in sp.truss_grid.crosssecs]
Ferrite.getnnodes(problem::StiffnessTopOptProblem) = Ferrite.getnnodes(getdh(problem).grid)

function TrussProblem(
    ::Type{Val{CellType}},
    node_points::Dict{iT,SVector{xdim,T}},
    elements::Dict{iT,Tuple{iT,iT}},
    loads::Dict{iT,SVector{xdim,T}},
    supports::Dict{iT,SVector{xdim,fT}},
    mats=TrussFEAMaterial{T}(1.0, 0.3),
    crosssecs=TrussFEACrossSec{T}(1.0),
) where {xdim,T,iT,fT,CellType}
    # unify number type
    # _T = promote_type(eltype(sizes), typeof(mats), typeof(ν), typeof(force))
    # if _T <: Integer
    #     T = Float64
    # else
    #     T = _T
    # end
    if CellType === :Linear
        truss_grid = TrussGrid(node_points, elements; crosssecs)
        geom_order = 1
    else
        @assert false "Other cell type not implemented"
    end
    # reference domain dimension for a line element
    ξdim = 1
    ncells = getncells(truss_grid)

    if mats isa Vector
        @assert length(mats) == ncells
        mats = convert(Vector{TrussFEAMaterial{T}}, mats)
    elseif mats isa TrussFEAMaterial
        mats = [convert(TrussFEAMaterial{T}, mats) for i in 1:ncells]
    else
        error("Invalid mats: $(mats)")
    end

    # * load nodeset
    # the grid node ordering coincides with the input node_points
    if haskey(truss_grid.grid.nodesets, "load")
        pop!(truss_grid.grid.nodesets, "load")
    end
    load_nodesets = Set{Int}()
    for (k, _) in loads
        push!(load_nodesets, k)
    end
    addnodeset!(truss_grid.grid, "load", load_nodesets)

    # * support nodeset
    for i in 1:xdim
        if haskey(truss_grid.grid.nodesets, get_fixities_node_set_name(i))
            pop!(truss_grid.grid.nodesets, get_fixities_node_set_name(i))
        end
        support_nodesets = Set{Int}()
        for (nodeidx, condition) in supports
            if condition[i]
                push!(support_nodesets, nodeidx)
            end
        end
        addnodeset!(truss_grid.grid, get_fixities_node_set_name(i), support_nodesets)
    end

    # Create displacement field u
    dh = DofHandler(truss_grid.grid)
    if CellType === :Linear
        # truss linear
        # interpolation_space
        ip = Lagrange{ξdim,RefCube,geom_order}()
        push!(dh, :u, xdim, ip)
    else
        # TODO truss 2-order
        @assert false "not implemented"
        # ip = Lagrange{2, RefCube, 2}()
        # push!(dh, :u, xdim, ip)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    for i in 1:xdim
        dbc = Dirichlet(
            :u,
            getnodeset(truss_grid.grid, get_fixities_node_set_name(i)),
            (x, t) -> zeros(T, 1),
            [i],
        )
        add!(ch, dbc)
    end
    close!(ch)

    # update the DBC to current time
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    # fnode = Tuple(getnodeset(truss_grid.grid, "load"))[1]
    # node_dofs = metadata.node_dofs
    # force_dof = node_dofs[2, fnode]

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return TrussProblem(truss_grid, mats, ch, loads, black, white, varind, metadata)
end

function Base.show(io::Base.IO, mime::MIME"text/plain", sp::TrussProblem)
    println(io, "TrussProblem:")
    print(io, "    ")
    Base.show(io, mime, sp.truss_grid)
    println(io, "    point loads: $(length(sp.force))")
    return println(io, "    active vars: $(sum(sp.varind .!= 0))")
end

#########################################

TopOpt.TopOptProblems.nnodespercell(p::TrussProblem) = nnodespercell(p.truss_grid)

"""
    getcloaddict(TrussProblem{xdim,T})

Get a dict (node_idx => force vector) for concentrated loads
"""
function TopOpt.TopOptProblems.getcloaddict(p::TrussProblem{xdim,T}) where {xdim,T}
    return p.force
end

function default_quad_order(::TrussProblem)
    return 1
end

#########################################

"""
    PointLoadCantileverTruss(nels::NTuple{dim,Int}, sizes::NTuple{dim}, E = 1.0, ν = 0.3, force = 1.0; k_connect=1) where {dim, CellType}

# Inputs

- `nels`: number of elements in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `sizes`: the size of each element in each direction, a 2-tuple for 2D problems and a 3-tuple for 3D problems
- `E`: Young's modulus
- `ν`: Poisson's ration
- `force`: force at the center right of the cantilever beam (positive is downward)
- `k_connect`: k-ring of a node to connect to form the ground mesh. Defaults to 1. For example, for a 2D domain, a node will be connected to `8` neighboring nodes if `k_connect=1`, and `8+16=24` neighboring nodes if `k_connect=2`.

# Returns
- `TrussProblem`

# Example
```
nels = (60,20);
sizes = (1.0,1.0);
E = 1.0;
ν = 0.3;
force = 1.0;
problem = PointLoadCantileverTruss(nels, sizes, E, ν, force, k_connect=2)
```
"""
function PointLoadCantileverTruss(
    nels::NTuple{dim,Int}, sizes::NTuple{dim}, E=1.0, ν=0.3, force=1.0; k_connect=1
) where {dim}
    iseven(nels[2]) && (length(nels) < 3 || iseven(nels[3])) ||
        throw("Grid does not have an even number of elements along the y and/or z axes.")
    _T = promote_type(eltype(sizes), typeof(E), typeof(ν), typeof(force))
    if _T <: Integer
        T = Float64
    else
        T = _T
    end

    # only for the convience of getting all the node points
    rect_grid = RectilinearGrid(Val{:Linear}, nels, T.(sizes))
    node_mat = hcat(map(x -> Vector(x.x), rect_grid.grid.nodes)...)
    kdtree = KDTree(node_mat)
    if dim == 2
        # 4+1*4 -> 4+3*4 -> 4+5*4
        k_ = 4 * k_connect + 4 * sum(1:2:(2 * k_connect - 1))
    else
        k_ = 8 * k_connect + 6 * sum(1:9:(9 * k_connect - 1))
    end
    idxs, _ = knn(kdtree, node_mat, k_ + 1, true)
    connect_mat = zeros(Int, 2, k_ * length(idxs))
    for (i, v) in enumerate(idxs)
        connect_mat[1, ((i - 1) * k_ + 1):(i * k_)] = ones(Int, k_) * i
        connect_mat[2, ((i - 1) * k_ + 1):(i * k_)] = v[2:end] # skip the point itself
    end
    truss_grid = TrussGrid(node_mat, connect_mat)

    # reference domain dimension for a line element
    ξdim = 1
    ncells = getncells(truss_grid)
    mats = [TrussFEAMaterial{T}(E, ν) for i in 1:ncells]

    # * support nodeset
    for i in 1:dim
        addnodeset!(truss_grid.grid, get_fixities_node_set_name(i), x -> left(rect_grid, x))
    end

    # * load nodeset
    if dim == 2
        addnodeset!(
            truss_grid.grid, "force", x -> right(rect_grid, x) && middley(rect_grid, x)
        )
    else
        addnodeset!(
            truss_grid.grid,
            "force",
            x -> right(rect_grid, x) && middley(rect_grid, x) && middlez(rect_grid, x),
        )
    end

    # * Create displacement field u
    geom_order = 1
    dh = DofHandler(truss_grid.grid)
    ip = Lagrange{ξdim,RefCube,geom_order}()
    push!(dh, :u, dim, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    for i in 1:dim
        dbc = Dirichlet(
            :u,
            getnodeset(truss_grid.grid, get_fixities_node_set_name(i)),
            (x, t) -> zeros(T, 1),
            [i],
        )
        add!(ch, dbc)
    end
    close!(ch)
    # update the DBC to current time
    t = T(0)
    update!(ch, t)

    metadata = Metadata(dh)

    loadset = getnodeset(truss_grid.grid, "force")
    ploads = Dict{Int,SVector{dim,T}}()
    for node_id in loadset
        ploads[node_id] = SVector{dim,T}(dim == 2 ? [0.0, force] : [0.0, 0.0, force])
    end

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)

    return TrussProblem(truss_grid, mats, ch, ploads, black, white, varind, metadata)
end
