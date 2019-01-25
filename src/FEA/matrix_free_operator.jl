abstract type AbstractMatrixOperator{Tconv} end

@params struct MatrixOperator{Tconv} <: AbstractMatrixOperator{Tconv}
    K
    f
    conv::Tconv
end
LinearAlgebra.mul!(c, op::MatrixOperator, b) = mul!(c, op.K, b)
Base.size(op::MatrixOperator, i...) = size(op.K, i...)
Base.eltype(op::MatrixOperator) = eltype(op.K)
LinearAlgebra.:*(op::MatrixOperator, b) = mul!(similar(b), op.K, b)

@params struct MatrixFreeOperator{Tconv, T, dim} <: AbstractMatrixOperator{Tconv}
    f::AbstractVector{T}
    elementinfo::ElementFEAInfo{dim, T}
    meandiag::T
    vars::AbstractVector{T}
    xes
    fixed_dofs
    free_dofs
    xmin::T
    penalty
    conv::Tconv
end
GPUUtils.whichdevice(m::MatrixFreeOperator) = whichdevice(m.vars)
Base.size(op::MatrixFreeOperator) = (size(op, 1), size(op, 2))
Base.size(op::MatrixFreeOperator, i) = 1 <= i <= 2 ? length(op.elementinfo.fixedload) : 1
Base.eltype(op::MatrixFreeOperator{<:Any, T}) where {T} = T

import LinearAlgebra: *, mul!

function *(A::MatrixFreeOperator, x)
    y = similar(x)
    mul!(y, A::MatrixFreeOperator, x)
    y
end

function mul!(y::TV, A::MatrixFreeOperator, x::TV) where {TV <: AbstractVector}
    T = eltype(y)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    dofspercell = size(A.elementinfo.Kes[1], 1)
    meandiag = A.meandiag

    @unpack Kes, metadata, black, white, varind = A.elementinfo
    @unpack cell_dofs, dof_cells = metadata
    @unpack penalty, xmin, vars, fixed_dofs, free_dofs, xes = A
    
    for i in 1:nels
        px = ifelse(black[i], one(T), 
                    ifelse(white[i], xmin, 
                            density(penalty(vars[varind[i]]), xmin)
                        )
                    )
        xe = xes[i]
        for j in 1:dofspercell
            xe = @set xe[j] = x[cell_dofs[j,i]]
        end
        if eltype(Kes) <: Symmetric
            xes[i] = px * (bcmatrix(Kes[i]).data * xe)
        else
            xes[i] = px * (bcmatrix(Kes[i]) * xe)
        end
    end

    for i in 1:length(fixed_dofs)
        dof = fixed_dofs[i]
        y[dof] = meandiag * x[dof]
    end
    for i in 1:length(free_dofs)
        dof = free_dofs[i]
        yi = zero(T)
        r = dof_cells.offsets[dof] : dof_cells.offsets[dof+1]-1
        for ind in r
            k, m = dof_cells.values[ind]
            yi += xes[k][m]
        end
        y[dof] = yi
    end
    y
end

function mul!(y::TV, A::MatrixFreeOperator, x::TV) where {TV <: CuArrays.CuVector}
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

    args2 = (y, x, dof_cells.offsets, dof_cells.values, xes, fixed_dofs, free_dofs, meandiag)
    callkernel(dev, mul_kernel2, args2)
    CUDAdrv.synchronize(ctx)

    return y
end

# CUDA kernels
function mul_kernel1(xes::AbstractVector{TV}, x, black, white, vars, varind, cell_dofs, Kes, xmin, penalty, nels) where {N, T, TV<:SVector{N, T}}
    i = @thread_global_index()
    offset = @total_threads()
    while i <= nels
        px = ifelse(black[i], one(T), 
                    ifelse(white[i], xmin, 
                            density(penalty(vars[varind[i]]), xmin)
                        )
                    )
        xe = xes[i]
        for j in 1:N
            xe = @set xe[j] = x[cell_dofs[j, i]]
        end
        if eltype(Kes) <: Symmetric
            xe = SVector{1, T}((px,)) .* (bcmatrix(Kes[i]).data * xe)
        else
            xe = SVector{1, T}((px,)) .* (bcmatrix(Kes[i]) * xe)
        end

        xes[i] = xe
        i += offset
    end

    return
end

function mul_kernel2(y, x, dof_cells_offsets, dof_cells_values, xes, fixed_dofs, free_dofs, meandiag)
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
        dof = free_dofs[i - n_fixeddofs]
        yi = zero(T)
        r = dof_cells_offsets[dof] : dof_cells_offsets[dof+1]-1
        for ind in r
            k, m = dof_cells_values[ind]
            yi += xes[k][m]
        end
        y[dof] = yi
        i += offset
    end
    return
end
