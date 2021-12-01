# TODO we should have an instruction for defining new problems
# module NewTopOptProblems
# export NewPointLoadCantilever

using Ferrite
using TopOpt
using TopOpt.TopOptProblems: RectilinearGrid, Metadata
using TopOpt.TopOptProblems: left, right, bottom, middley, middlez,
    nnodespercell, nfacespercell, find_black_and_white, find_varind
using TopOpt.Utilities: @params

@params struct NewPointLoadCantilever{dim, T, N, M} <: StiffnessTopOptProblem{dim, T}
    rect_grid::RectilinearGrid{dim, T, N, M}
    E::T
    ν::T
    ch::ConstraintHandler{<:DofHandler{dim, <:Cell{dim,N,M}, T}, T}
    load_dict::Dict{Int, Vector{T}}
    black::AbstractVector
    white::AbstractVector
    varind::AbstractVector{Int}
    metadata::Metadata
end
    # force::T
    # force_dof::Integer

function NewPointLoadCantilever(::Type{Val{CellType}}, nels::NTuple{dim,Int}, sizes::NTuple{dim}, 
    E = 1.0, ν = 0.3, force = 1.0) where {dim, CellType}
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
    if dim == 3
        addnodeset!(rect_grid.grid, "down_force", x -> right(rect_grid, x) && 
            bottom(rect_grid, x));
            #  && middlez(rect_grid, x));
    else
        addnodeset!(rect_grid.grid, "down_force", x -> right(rect_grid, x) && 
            right(rect_grid, x) && middley(rect_grid, x));
    end

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
    Ferrite.update!(ch, t)

    metadata = Metadata(dh)
    load_dict = Dict{Int, Vector{T}}()
    for fnode in getnodeset(rect_grid.grid, "down_force")
    	load_dict[fnode] = [0, -force, 0]
    end

    N = nnodespercell(rect_grid)
    M = nfacespercell(rect_grid)

    black, white = find_black_and_white(dh)
    varind = find_varind(black, white)
    
    return NewPointLoadCantilever(rect_grid, E, ν, ch, load_dict, black, white, varind, metadata)
end

# used in FEA to determine default quad order
# we don't assume the problem struct has `rect_grid` to define its grid
TopOptProblems.nnodespercell(p::NewPointLoadCantilever) = nnodespercell(p.rect_grid)

# ! important, used for specification!
function TopOptProblems.getcloaddict(p::NewPointLoadCantilever{dim, T}) where {dim, T}
    # f = T[0, -p.force, 0]
    # fnode = Tuple(getnodeset(p.rect_grid.grid, "down_force"))[1]
    # return Dict{Int, Vector{T}}(fnode => f)
    return p.load_dict
end

# end # end NewTopOptProblems module