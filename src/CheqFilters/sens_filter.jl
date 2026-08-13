"""
    SensFilterFun(solver; rmin)

Sensitivity chequerboard filter with radius `rmin`. Smooths the objective
gradient by weighting each element's sensitivity with the sensitivities of
neighboring elements within `rmin`, using the filter scheme from
[HuangXie2010](@cite) (BESO): the weight is `max(rmin - dist, 0)` where
`dist` is the distance from the element centroid to the neighboring node.

Call as `y = flt(x)`. See also [BendsoeSigmund2003](@cite) §3.4 for general
background on sensitivity filtering.
"""
struct SensFilterFun{T,TV<:AbstractVector{T},TE<:ElementFEAInfo,TM<:FilterMetadata} <:
       AbstractSensFilter
    elementinfo::TE
    metadata::TM
    rmin::T
    nodal_grad::TV
    last_grad::TV
    cell_weights::TV
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::SensFilterFun)
    return println(io, "TopOpt sensitivity filter")
end

function SensFilterFun(solver::AbstractFEASolver; rmin)
    return SensFilterFun(solver, rmin)
end
function SensFilterFun(
    solver::TS, rmin::T, (::Type{TI})=Int
) where {T,TI<:Integer,TS<:AbstractFEASolver}
    metadata = FilterMetadata(solver, rmin, TI)
    TM = typeof(metadata)
    problem = solver.problem
    elementinfo = solver.elementinfo
    grid = problem.ch.dh.grid
    nnodes = getnnodes(grid)
    nodal_grad = zeros(T, nnodes)

    nel = getncells(grid)
    # last_grad stores filtered sensitivities (same length as design variables)
    last_grad = zeros(T, nel)

    cell_weights = zeros(T, nnodes)

    return SensFilterFun(elementinfo, metadata, rmin, nodal_grad, last_grad, cell_weights)
end

function (cf::SensFilterFun)(x::PseudoDensities{I,P}) where {I,P}
    return PseudoDensities{I,P,true}(x.x)
end
function ChainRulesCore.rrule(cf::SensFilterFun, x::PseudoDensities)
    return x,
    Δ -> begin
        Δ = ChainRulesCore.unthunk(Δ)
        if hasproperty(Δ, :x)
            newΔ = copy(Δ.x)
        else
            newΔ = copy(Δ)
        end
        @unpack elementinfo, nodal_grad, cell_weights, metadata = cf
        @unpack cellvolumes, cells = elementinfo
        @unpack cell_neighbouring_nodes, cell_node_weights = metadata
        node_cells = elementinfo.metadata.node_cells
        update_nodal_grad!(nodal_grad, node_cells, cell_weights, cells, cellvolumes, newΔ)
        normalize_grad!(nodal_grad, cell_weights)
        update_grad!(newΔ, cell_neighbouring_nodes, cell_node_weights, nodal_grad)
        return (NoTangent(), Tangent{typeof(x)}(; x=newΔ))
    end
end

function update_nodal_grad!(
    nodal_grad::AbstractVector, node_cells, cell_weights, cells, cellvolumes, grad
)
    T = eltype(nodal_grad)
    for n in eachindex(nodal_grad)
        nodal_grad[n] = zero(T)
        cell_weights[n] = zero(T)
        r = node_cells.offsets[n]:(node_cells.offsets[n + 1] - 1)
        for i in r
            c = node_cells.values[i][1]
            w = cellvolumes[c]
            cell_weights[n] += w
            nodal_grad[n] += w * grad[c]
        end
    end
    return nodal_grad
end

function normalize_grad!(nodal_grad::AbstractVector, cell_weights)
    for n in eachindex(nodal_grad)
        if cell_weights[n] > 0
            nodal_grad[n] /= cell_weights[n]
        end
    end
end

function update_grad!(
    grad::AbstractVector, cell_neighbouring_nodes, cell_node_weights, nodal_grad
)
    for i in 1:(length(cell_neighbouring_nodes.offsets) - 1)
        nodes = cell_neighbouring_nodes[i]
        if length(nodes) == 0
            continue
        end
        weights = cell_node_weights[i]
        grad[i] = dot(view(nodal_grad, nodes), weights) / sum(weights)
    end

    return nothing
end
