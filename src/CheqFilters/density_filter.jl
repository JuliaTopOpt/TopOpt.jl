struct DensityFilter{T,TM<:FilterMetadata,TJ<:AbstractMatrix{T}} <:
       AbstractDensityFilter
    metadata::TM
    rmin::T
    jacobian::TJ
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::DensityFilter)
    return println(io, "TopOpt density filter")
end
Nonconvex.NonconvexCore.getdim(f::DensityFilter) = size(f.jacobian, 1)

DensityFilter(solver; rmin) = DensityFilter(solver, rmin)
function DensityFilter(
    solver::TS, rmin::T, (::Type{TI})=Int
) where {T,TI<:Integer,TS<:AbstractFEASolver}
    metadata = FilterMetadata(solver, rmin, TI)
    TM = typeof(metadata)
    
    jacobian = getJacobian(solver, metadata)
    return DensityFilter(metadata, rmin, jacobian)
end

function (cf::DensityFilter)(x::PseudoDensities{I,P}) where {I,P}
    @unpack jacobian = cf
    out = similar(x.x)
    mul!(out, jacobian, x.x)
    return PseudoDensities{I,P,true}(out)
end
function ChainRulesCore.rrule(f::DensityFilter, x::PseudoDensities)
    return f(x), Δ -> begin
        _Δ = hasproperty(Δ, :x) ? Δ.x : Δ
        (nothing, Tangent{typeof(x)}(; x=f.jacobian' * _Δ))
    end
end

function getJacobian(solver, metadata::FilterMetadata)
    @unpack elementinfo, problem = solver
    @unpack cellvolumes = elementinfo
    @unpack cell_neighbouring_nodes, cell_node_weights = metadata
    node_cells = elementinfo.metadata.node_cells

    T = eltype(cellvolumes)
    grid = problem.ch.dh.grid
    nnodes = getnnodes(grid)
    nel = length(cellvolumes)
    I = Int[]
    J = Int[]
    V = T[]
    for n in 1:nnodes
        r = node_cells.offsets[n]:(node_cells.offsets[n + 1] - 1)
        for i in r
            c = node_cells.values[i][1]
            w = cellvolumes[c]
            push!(I, n)
            push!(J, c)
            push!(V, w)
        end
    end
    mat1_transpose = sparse(J, I, V, nel, nnodes)
    scalecols!(mat1_transpose)
    norm(V) == 0 && throw("Jacobian is all 0s.")

    I = Int[]
    J = Int[]
    V = T[]
    for i in 1:nel
        nodes = cell_neighbouring_nodes[i]
        if length(nodes) == 0
            continue
        end
        weights = cell_node_weights[i]
        weights_sum = sum(weights)
        for (j, n) in enumerate(nodes)
            push!(I, i)
            push!(J, n)
            push!(V, weights[j] / weights_sum)
        end
    end
    mat2_transpose = sparse(J, I, V, nnodes, nel)
    norm(V) == 0 && throw("Jacobian is all 0s.")
    mat_transpose = mat1_transpose * mat2_transpose
    return mat_transpose'
end

function scalecols!(A::SparseMatrixCSC)
    @unpack colptr, nzval = A
    T = eltype(A)
    for col in 1:(length(colptr) - 1)
        inds = colptr[col]:(colptr[col + 1] - 1)
        s = sum(nzval[inds])
        if s != 0
            nzval[inds] .= nzval[inds] ./ s
        end
    end
    return A
end

struct ProjectedDensityFilter{TF<:DensityFilter,TP1,TP2} <: AbstractDensityFilter
    filter::TF
    preproj::TP1
    postproj::TP2
end
function Nonconvex.NonconvexCore.getdim(f::ProjectedDensityFilter)
    return Nonconvex.NonconvexCore.getdim(f.filter)
end
function (cf::ProjectedDensityFilter)(x::PseudoDensities{I,P}) where {I,P}
    if cf.preproj isa Nothing
        fx = x.x
    else
        fx = cf.preproj.(x.x)
    end
    fx = cf.filter(PseudoDensities{I,P,true}(fx)).x
    if cf.postproj isa Nothing
        out = fx
    else
        out = cf.postproj.(fx)
    end
    return PseudoDensities{I,P,true}(out)
end
