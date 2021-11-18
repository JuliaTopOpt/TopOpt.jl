using ..TopOpt: GPU, @init_cuda
import ..TopOpt: whichdevice
import ..CUDASupport
using ..GPUUtils

@init_cuda

@define_cu(FilterMetadata, :cell_neighbouring_nodes, :cell_node_weights)
whichdevice(m::FilterMetadata) = whichdevice(m.cell_neighbouring_nodes)

@define_cu(SensFilter, :metadata, :nodal_grad, :last_grad, :cell_weights)
whichdevice(c::SensFilter) = whichdevice(c.nodal_grad)

function update_nodal_grad!(nodal_grad::CuVector, node_cells, args...)
    T = eltype(nodal_grad)
    allargs = (nodal_grad, node_cells.offsets, node_cells.values, args...)
    callkernel(dev, cheq_kernel1, allargs)
    CUDAdrv.synchronize(ctx)
    return
end

function cheq_kernel1(
    nodal_grad,
    node_cells_offsets,
    node_cells_values,
    cell_weights,
    cells,
    cellvolumes,
    black,
    white,
    varind,
    grad,
)
    T = eltype(nodal_grad)
    n = @thread_global_index()
    offset = @total_threads()
    while n <= length(nodal_grad)
        nodal_grad[n] = zero(T)
        cell_weights[n] = zero(T)
        r = node_cells_offsets[n]:node_cells_offsets[n+1]-1
        for i in r
            c = node_cells_values[i][1]
            if black[c] || white[c]
                continue
            end
            ind = varind[c]
            w = cellvolumes[c]
            cell_weights[n] += w
            nodal_grad[n] += w * grad[ind]
        end
        n += offset
    end
end

function normalize_grad!(nodal_grad::CuVector, cell_weights)
    T = eltype(nodal_grad)
    args = (nodal_grad, cell_weights)
    callkernel(dev, cheq_kernel2, args)
    CUDAdrv.synchronize(ctx)
    return
end
function cheq_kernel2(nodal_grad, cell_weights)
    T = eltype(nodal_grad)
    n = @thread_global_index()
    offset = @total_threads()
    while n <= length(nodal_grad)
        w = cell_weights[n]
        w = ifelse(w > 0, w, one(T))
        nodal_grad[n] /= w
        n += offset
    end
    return
end

function update_grad!(
    grad::CuVector,
    black,
    white,
    varind,
    cell_neighbouring_nodes,
    cell_node_weights,
    nodal_grad,
)
    T = eltype(grad)
    allargs = (
        grad,
        black,
        white,
        varind,
        cell_neighbouring_nodes.offsets,
        cell_neighbouring_nodes.values,
        cell_node_weights.values,
        nodal_grad,
    )
    callkernel(dev, cheq_kernel3, allargs)
    CUDAdrv.synchronize(ctx)
    return
end
function cheq_kernel3(
    grad,
    black,
    white,
    varind,
    cell_neighbouring_nodes_offsets,
    cell_neighbouring_nodes_values,
    cell_node_weights_values,
    nodal_grad,
)
    T = eltype(nodal_grad)
    i = @thread_global_index()
    offset = @total_threads()
    while i <= length(black)
        if black[i] || white[i]
            continue
        end
        ind = varind[i]
        r = cell_neighbouring_nodes_offsets[ind]:cell_neighbouring_nodes_offsets[ind+1]-1
        length(r) == 0 && continue
        grad[ind] = zero(T)
        sum_weights = zero(T)
        for linear_ind in r
            w = cell_node_weights_values[linear_ind]
            sum_weights += w
            node_ind = cell_neighbouring_nodes_values[linear_ind]
            grad[ind] += nodal_grad[node_ind] * w
        end
        grad[ind] /= sum_weights
        i += offset
    end

    return
end
