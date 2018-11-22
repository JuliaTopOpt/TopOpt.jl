struct MatrixFreeOperator{T, dim, TEInfo<:ElementFEAInfo{dim, T}, TS<:StiffnessTopOptProblem{dim, T}, Tf<:AbstractVector{T}, TP}
    elementinfo::TEInfo
    meandiag::T
    problem::TS
    vars::Tf
    xmin::T
    penalty::TP
end
Base.size(op::MatrixFreeOperator) = (size(op, 1), size(op, 2))
Base.size(op::MatrixFreeOperator, i) = 1 <= i <= 2 ? length(op.elementinfo.fixedload) : 1
Base.eltype(op::MatrixFreeOperator{T}) where {T} = T

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

    @unpack Kes, fes, metadata, black, white, varind = A.elementinfo
    @unpack cell_dofs, dof_cells, dof_cells_offset = metadata
    @unpack penalty, xmin, vars = A
    
    for i in 1:nels
        px = ifelse(black[i], one(T), 
                    ifelse(white[i], xmin, 
                            density(penalty(vars[varind[i]]), xmin)
                        )
                    )
        fe = fes[i]
        for j in 1:dofspercell
            fe = @set fe[j] = x[cell_dofs[j,i]]
        end
        if eltype(Kes) <: Symmetric
            fes[i] = px * (Kes[i].data * fe)
        else
            fes[i] = px * (Kes[i] * fe)
        end
    end

    for i in 1:ndofs
        yi = zero(T)
        r = dof_cells_offset[i] : dof_cells_offset[i+1]-1
        for ind in r
            k, m = dof_cells[ind]
            yi += fes[k][m]
        end
        y[i] = yi
    end
    y
end

function mul!(y::TV, A::MatrixFreeOperator, x::TV) where {TV <: CuArrays.CuVector}
    T = eltype(y)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    dofspercell = size(A.elementinfo.Kes[1], 1)
    
    @unpack Kes, fes, metadata, black, white, varind = A.elementinfo
    @unpack cell_dofs, dof_cells, dof_cells_offset = metadata
    @unpack penalty, xmin, vars = A

    dev = CUDAdrv.device()
    ctx = CUDAdrv.CuContext(dev)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threads = min(nels, MAX_THREADS_PER_BLOCK)
    blocks = ceil.(Int, nels / threads)
    @cuda blocks=blocks threads=threads kernel1(x, black, white, vars, varind, cell_dofs, Kes, fes, xmin, penalty, nels)

    CUDAdrv.synchronize(ctx)

    threads = min(ndofs, MAX_THREADS_PER_BLOCK)
    blocks = ceil.(Int, ndofs / threads)
    @cuda blocks=blocks threads=threads kernel2(y, dof_cells_offset, dof_cells, fes, ndofs)
    y
end

# CUDA kernels
function kernel1(x, black, white, vars, varind, cell_dofs, Kes, fes::AbstractVector{TV}, xmin, penalty, nels) where {N, T, TV<:SVector{N, T}}
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if i <= nels
        px = vars[varind[i]]
        #px = ifelse(black[i], one(T), 
        #            ifelse(white[i], xmin, 
        #                    density(penalty(vars[varind[i]]), xmin)
        #                )
        #            )
        fe = fes[i]
        for j in 1:N
            fe = @set fe[j] = x[cell_dofs[j,i]]
        end
        if eltype(Kes) <: Symmetric
            fe = SVector{1, T}((px,)) .* (Kes[i].data * fe)
        else
            fe = SVector{1, T}((px,)) .* (Kes[i] * fe)
        end
        
        fes[i] = fe
    end

    return
end

function kernel2(y, dof_cells_offset, dof_cells, fes, ndofs)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    T = eltype(y)
    if i <= ndofs
        yi = zero(T)
        r = dof_cells_offset[i] : dof_cells_offset[i+1]-1
        for ind in r
            k, m = dof_cells[ind]
            yi += fes[k][m]
        end
        y[i] = yi
    end
    return
end
