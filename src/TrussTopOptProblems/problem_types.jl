using ..TopOpt.TopOptProblems: StiffnessTopOptProblem, Metadata

get_fixities_node_set_name(i) = "fixed_u$(i)"

# @params
struct TrussProblem{xdim,T,N,M} <: StiffnessTopOptProblem{xdim,T}
    truss_grid::TrussGrid{xdim,T,N,M} # ground truss mesh
    materials::Vector{TrussFEAMaterial{T}}
    ch::ConstraintHandler{<:DofHandler{xdim,<:Ferrite.Cell{xdim,N,M},T},T}
    force::Dict{Int, SVector{xdim, T}}
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int} # variable dof => free dof, based on black & white
    metadata::Metadata
end
# - `force_dof`: dof number at which the force is applied

gettrussgrid(sp::TrussProblem) = sp.truss_grid
getFerritegrid(sp::TrussProblem) = sp.truss_grid.grid

TopOpt.TopOptProblems.getE(sp::TrussProblem) = [m.E for m in sp.materials]
TopOpt.TopOptProblems.getν(sp::TrussProblem) = [m.ν for m in sp.materials]
getA(sp::TrussProblem) = [cs.A for cs in sp.truss_grid.crosssecs]
Ferrite.getnnodes(problem::StiffnessTopOptProblem) = Ferrite.getnnodes(getdh(problem).grid)

function TrussProblem(::Type{Val{CellType}}, node_points::Dict{iT, SVector{xdim, T}}, elements::Dict{iT, Tuple{iT, iT}}, 
    loads::Dict{iT, SVector{xdim, T}}, supports::Dict{iT, SVector{xdim, fT}}, mats=TrussFEAMaterial{T}(1.0, 0.3), crosssecs=TrussFEACrossSec{T}(1.0)) where {xdim, T, iT, fT, CellType}
    # unify number type
    # _T = promote_type(eltype(sizes), typeof(mats), typeof(ν), typeof(force))
    # if _T <: Integer
    #     T = Float64
    # else
    #     T = _T
    # end
    if CellType === :Linear
        # TODO load should be added here as well
        truss_grid = TrussGrid(node_points, elements, supports; crosssecs)
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
        mats = [TrussFEAMaterial{T}(mats) for i=1:ncells]
    else
        error("Invalid mats: $(mats)")
    end

    # * load nodeset
    # the grid node ordering coincides with the input node_points
    if haskey(truss_grid.grid.nodesets, "load")
        pop!(truss_grid.grid.nodesets, "load")
    end
    load_nodesets = Set{Int}()
    for (k,_) in loads
        push!(load_nodesets, k)
    end
    addnodeset!(truss_grid.grid, "load", load_nodesets)

    # * support nodeset
    for i=1:xdim
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
        ip = Lagrange{ξdim, RefCube, geom_order}()
        push!(dh, :u, xdim, ip)
    else
        # TODO truss 2-order
        @assert false "not implemented"
        # ip = Lagrange{2, RefCube, 2}()
        # push!(dh, :u, xdim, ip)
    end
    close!(dh)

    ch = ConstraintHandler(dh)
    for i=1:xdim
        dbc = Dirichlet(:u, getnodeset(truss_grid.grid, get_fixities_node_set_name(i)), (x,t)->T[0], [i])
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
    println(io, "    E: $(sp.E)")
    println(io, "    point loads: $(length(sp.force))")
    println(io, "    active vars: $(sum(sp.varind .!= 0))")
end

#########################################

TopOpt.TopOptProblems.nnodespercell(p::TrussProblem) = nnodespercell(p.truss_grid)

"""
    getcloaddict(TrussProblem{xdim,T})

Get a dict (node_idx => force vector) for concentrated loads
"""
function TopOpt.TopOptProblems.getcloaddict(p::TrussProblem{xdim,T}) where {xdim, T}
    return p.force
end

function default_quad_order(::TrussProblem)
    return 1
end

getξdim(::TrussProblem) = 1

#######################################
# * extra Cell types for Line elements
# https://github.com/lijas/Ferrite.jl/blob/line2/src/Grid/grid.jl

# const Line2d = Cell{2,2,2}
# const Line3d = Cell{3,2,2}
# const QuadraticLine = Cell{1,3,2}

# # 1D: vertices, Line is defined in Ferrite
# Ferrite.faces(c::Union{QuadraticLine}) = (c.nodes[1], c.nodes[2])
# Ferrite.vertices(c::Union{Line2d,Line3d,QuadraticLine}) = (c.nodes[1], c.nodes[2])

# # 2D: vertices, faces
# Ferrite.faces(c::Line2d) = ((c.nodes[1],c.nodes[2]),) 

# # 3D: vertices, edges, faces
# Ferrite.edges(c::Line3d) = ((c.nodes[1],c.nodes[2]),) 

# Ferrite.default_interpolation(::Union{Type{Line},Type{Line2d},Type{Line3d}}) = Lagrange{1,RefCube,1}()