@params struct DensityFilter{_filtering, T} <: AbstractDensityFilter
    filtering::Val{_filtering}
    metadata::FilterMetadata
    rmin::T
    jacobian::AbstractMatrix{T}
end
TopOpt.dim(f::DensityFilter) = size(f.jacobian, 1)
DensityFilter{true}(args...) = DensityFilter(Val(true), args...)
DensityFilter{false}(args...) = DensityFilter(Val(false), args...)

function DensityFilter(::Val{filtering}, solver::AbstractFEASolver, args...) where {filtering}
    DensityFilter(Val(filtering), whichdevice(solver), solver, args...)
end

function DensityFilter(::Val{true}, ::CPU, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TI<:Integer, TS<:AbstractFEASolver}
    metadata = FilterMetadata(solver, rmin, TI)
    TM = typeof(metadata)
    problem = solver.problem
    grid = problem.ch.dh.grid
    nnodes = getnnodes(grid)

    black = problem.black
    white = problem.white
    nel = length(black)
    nfc = sum(black) + sum(white)
    jacobian = getJacobian(solver, metadata)
    return DensityFilter(Val(true), metadata, rmin, jacobian)
end

function DensityFilter(::Val{false}, ::CPU, solver::TS, rmin::T, ::Type{TI}=Int) where {T, TS<:AbstractFEASolver, TI<:Integer}
    metadata = FilterMetadata(T, TI)
    cell_weights = T[]
    jacobian = T[]
    return DensityFilter(Val(false), metadata, rmin, jacobian)
end

function (cf::DensityFilter{true, T})(x) where {T}
    cf.rmin <= 0 && return x
    @unpack jacobian = cf
    out = similar(x)
    mul!(out, jacobian, x)
    return out
end
function TopOpt.jtvp!(out, cf::DensityFilter{true}, x, w; runf=true)
    mul!(out, cf.jacobian', w)
    return out
end

(cf::DensityFilter{false})(x) = x
function TopOpt.jtvp!(out, cf::DensityFilter{false}, x, w; runf=true)
    out .= w
    return out
end

function getJacobian(solver, metadata::FilterMetadata)
    @unpack elementinfo, problem = solver
    @unpack black, white, varind, cellvolumes = elementinfo
    @unpack cell_neighbouring_nodes, cell_node_weights = metadata
    node_cells = elementinfo.metadata.node_cells

    T = eltype(cellvolumes)
    grid = problem.ch.dh.grid
    nnodes = getnnodes(grid)
    I = Int[]
    J = Int[]
    V = T[]
    for n in 1:nnodes
        r = node_cells.offsets[n]:node_cells.offsets[n+1]-1
        for i in r
            c = node_cells.values[i][1]
            if black[c] || white[c]
                continue
            end
            ind = varind[c]
            w = cellvolumes[c]
            push!(I, n)
            push!(J, ind)
            push!(V, w)
        end
    end
    mat1_transpose = sparse(J, I, V, length(solver.vars), nnodes)
    scalecols!(mat1_transpose)
    norm(V) == 0 && throw("Jacobian is all 0s.")

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:length(black)
        if black[i] || white[i]
            continue
        end
        ind = varind[i]
        nodes = cell_neighbouring_nodes[ind]
        if length(nodes) == 0
            continue
        end
        weights = cell_node_weights[ind]
        weights_sum = sum(weights)
        for (j, n) in enumerate(nodes)
            push!(I, ind)
            push!(J, n)
            push!(V, weights[j] / weights_sum)
        end
    end
    mat2_transpose = sparse(J, I, V, nnodes, length(solver.vars))
    norm(V) == 0 && throw("Jacobian is all 0s.")
    mat_transpose = mat1_transpose * mat2_transpose
    return mat_transpose'
end

function scalecols!(A::SparseMatrixCSC)
    @unpack colptr, nzval = A
    T = eltype(A)
    for col in 1:length(colptr)-1
        inds = colptr[col]:colptr[col+1]-1
        s = sum(nzval[inds])
        if s != 0
            nzval[inds] .= nzval[inds] ./ s
        end
    end
    return A
end

@params struct ProjectedDensityFilter <: AbstractDensityFilter
    filter::DensityFilter
    preproj
    postproj
end
TopOpt.dim(f::ProjectedDensityFilter) = TopOpt.dim(f.filter)
function (cf::ProjectedDensityFilter)(x)
    if cf.preproj isa Nothing
        fx = x
    else
        fx = cf.preproj.(x)
    end
    fx = cf.filter(fx)
    if cf.postproj isa Nothing
        out = fx
    else
        out = cf.postproj.(fx)
    end
    return out
end
function TopOpt.jtvp!(out, cf::ProjectedDensityFilter, x, w; runf=true)
    if cf.preproj isa Nothing
        fx = cf.filter(x)
    else
        fx = cf.filter(cf.preproj.(x))
    end
    if cf.postproj isa Nothing
        w2 = w
    else
        w2 = ForwardDiff.derivative.(Ref(cf.postproj), fx) .* w
    end
    TopOpt.jtvp!(out, cf.filter, x, w2; runf=false)
    if !(cf.preproj isa Nothing)
        out .= ForwardDiff.derivative.(Ref(cf.preproj), x) .* out
    end
    return out
end
