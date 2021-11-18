using ..CUDASupport
using ..TopOpt: @init_cuda
@init_cuda()

using ..GPUUtils
import ..TopOpt: whichdevice

whichdevice(s::AbstractDisplacementSolver) = whichdevice(s.u)

for T in (:DirectDisplacementSolver, :PCGDisplacementSolver)
    @eval @inline CuArrays.cu(p::$T) = error("$T does not support the GPU.")
end

function update_f!(
    f::CuVector{T},
    values,
    prescribed_dofs,
    applyzero,
    dof_cells,
    cell_dofs,
    black,
    white,
    Kes,
    xmin,
    penalty,
    vars,
    varind,
    M,
) where {T}
    args = (
        f,
        values,
        prescribed_dofs,
        applyzero,
        dof_cells.offsets,
        dof_cells.values,
        cell_dofs,
        black,
        white,
        Kes,
        xmin,
        penalty,
        vars,
        varind,
        M,
    )
    callkernel(dev, bc_kernel, args)
    CUDAdrv.synchronize(ctx)
    return
end

function bc_kernel(
    f::AbstractVector{T},
    values,
    prescribed_dofs,
    applyzero,
    dof_cells_offsets,
    dof_cells_values,
    cell_dofs,
    black,
    white,
    Kes,
    xmin,
    penalty,
    vars,
    varind,
    M,
) where {T}

    ind = @thread_global_index()
    offset = @total_threads()
    while ind <= length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(Kes[ind], 1)

        r = dof_cells_offsets[d]:dof_cells_offsets[d+1]-1
        if !applyzero && v != 0
            for idx in r
                (i, j) = dof_cells_values[idx]
                if PENALTY_BEFORE_INTERPOLATION
                    px = ifelse(
                        black[i],
                        one(T),
                        ifelse(white[i], xmin, density(penalty(vars[varind[i]]), xmin)),
                    )
                else
                    px = ifelse(
                        black[i],
                        one(T),
                        ifelse(white[i], xmin, penalty(density(vars[varind[i]], xmin))),
                    )
                end
                if eltype(Kes) <: Symmetric
                    Ke = Kes[i].data
                else
                    Ke = Kes[i]
                end
                for row = 1:m
                    f[cell_dofs[row, i]] -= px * v * Ke[row, j]
                end
            end
        end
        f[d] = M * v
        ind += offset
    end
    return
end

@define_cu(
    StaticMatrixFreeDisplacementSolver,
    :f,
    :problem,
    :vars,
    :cg_statevars,
    :elementinfo,
    :penalty,
    :prev_penalty,
    :u,
    :fixed_dofs,
    :free_dofs,
    :xes,
    :lhs,
    :rhs
)

whichdevice(m::MatrixFreeOperator) = whichdevice(m.vars)

function mul!(y::TV, A::MatrixFreeOperator, x::TV) where {TV<:CuArrays.CuVector}
    T = eltype(y)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    meandiag = A.meandiag

    @unpack Kes, metadata, black, white, varind = A.elementinfo
    @unpack cell_dofs, dof_cells = metadata
    @unpack penalty, xmin, vars, fixed_dofs, free_dofs, xes = A

    args1 = (xes, x, black, white, vars, varind, cell_dofs, Kes, xmin, penalty, nels)
    callkernel(dev, mul_kernel1, args1)
    CUDAdrv.synchronize(ctx)

    args2 =
        (y, x, dof_cells.offsets, dof_cells.values, xes, fixed_dofs, free_dofs, meandiag)
    callkernel(dev, mul_kernel2, args2)
    CUDAdrv.synchronize(ctx)

    return y
end

# CUDA kernels
function mul_kernel1(
    xes::AbstractVector{TV},
    x,
    black,
    white,
    vars,
    varind,
    cell_dofs,
    Kes,
    xmin,
    penalty,
    nels,
) where {N,T,TV<:SVector{N,T}}
    i = @thread_global_index()
    offset = @total_threads()
    while i <= nels
        if PENALTY_BEFORE_INTERPOLATION
            px = ifelse(
                black[i],
                one(T),
                ifelse(white[i], xmin, density(penalty(vars[varind[i]]), xmin)),
            )
        else
            px = ifelse(
                black[i],
                one(T),
                ifelse(white[i], xmin, penalty(density(vars[varind[i]], xmin))),
            )
        end
        xe = xes[i]
        for j = 1:N
            xe = @set xe[j] = x[cell_dofs[j, i]]
        end
        if eltype(Kes) <: Symmetric
            xe = SVector{1,T}((px,)) .* (bcmatrix(Kes[i]).data * xe)
        else
            xe = SVector{1,T}((px,)) .* (bcmatrix(Kes[i]) * xe)
        end

        xes[i] = xe
        i += offset
    end

    return
end

function mul_kernel2(
    y,
    x,
    dof_cells_offsets,
    dof_cells_values,
    xes,
    fixed_dofs,
    free_dofs,
    meandiag,
)
    T = eltype(y)
    offset = @total_threads()
    n_fixeddofs = length(fixed_dofs)
    ndofs = length(y)

    i = @thread_global_index()
    while i <= n_fixeddofs
        dof = fixed_dofs[i]
        y[dof] = meandiag * x[dof]
        i += offset
    end

    i = @thread_global_index()
    while n_fixeddofs < i <= ndofs
        dof = free_dofs[i-n_fixeddofs]
        yi = zero(T)
        r = dof_cells_offsets[dof]:dof_cells_offsets[dof+1]-1
        for ind in r
            k, m = dof_cells_values[ind]
            yi += xes[k][m]
        end
        y[dof] = yi
        i += offset
    end
    return
end

whichdevice(r::LinearElasticityResult) = whichdevice(r.u)

function fill_vars!(vars::GPUArray, topology; round)
    if round
        vars .= round.(typeof(vars)(topology))
    else
        copyto!(vars, topology)
    end
    return vars
end
