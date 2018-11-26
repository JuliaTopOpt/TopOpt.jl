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

    args1 = (fes, x, black, white, vars, varind, cell_dofs, Kes, xmin, penalty, nels)
    callkernel(dev, mul_kernel1, args1)
    CUDAdrv.synchronize(ctx)
    
    args2 = (y, dof_cells_offset, dof_cells, fes, ndofs)
    callkernel(dev, mul_kernel2, args2)
    CUDAdrv.synchronize(ctx)

    return y
end

function callkernel(dev, kernel, args)
    blocks, threads = getvalidconfig(dev, kernel, args)
    @show blocks, threads
    @cuda blocks=blocks threads=threads kernel(args...)

    return
end

function getvalidconfig(dev, kernel, parallel_args)
    R = parallel_args[1]
    Rlength = length(R)
    Ssize = size(R)
    Slength = prod(Ssize)
    GC.@preserve parallel_args begin
        parallel_kargs = cudaconvert.(parallel_args)
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(kernel, parallel_tt)

        # we are limited in how many threads we can launch...
        ## by the kernel
        kernel_threads = CUDAnative.maxthreads(parallel_kernel)
        ## by the device
        block_threads = (x=CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_X),
                         y=CUDAdrv.attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Y),
                         total=CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))

        # figure out a legal launch configuration
        y_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, block_threads.y, kernel_threads)
        x_thr = min(512 ÷ y_thr, Slength, block_threads.x,
                    ceil(Int, block_threads.total/y_thr),
                    ceil(Int, kernel_threads/y_thr))

        blk, thr = (Rlength - 1) ÷ y_thr + 1, (x_thr, y_thr, 1)
        blk = min(blk, ceil(Int, Rlength / prod(thr)))
    end
    return blk, thr
end

# CUDA kernels
function mul_kernel1(fes::AbstractVector{TV}, x, black, white, vars, varind, cell_dofs, Kes, xmin, penalty, nels) where {N, T, TV<:SVector{N, T}}
    #blockid = blockIdx().x + blockIdx().y * gridDim().x
    #i = (blockid - 1) * (blockDim().x * blockDim().y) + (threadIdx().y * blockDim().x) + threadIdx().x
    i = thread_global_index()
    offset = total_threads()
    @inbounds while i <= nels
        #px = vars[varind[i]]
        px = ifelse(black[i], one(T), 
                    ifelse(white[i], xmin, 
                            density(penalty(vars[varind[i]]), xmin)
                        )
                    )
        fe = fes[i]
        for j in 1:N
            fe = @set fe[j] = x[cell_dofs[j, i]]
        end
        if eltype(Kes) <: Symmetric
            fe = SVector{1, T}((px,)) .* (Kes[i].data * fe)
        else
            fe = SVector{1, T}((px,)) .* (Kes[i] * fe)
        end

        fes[i] = fe
        i += offset
    end

    return
end

function mul_kernel2(y, dof_cells_offset, dof_cells, fes, ndofs)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    T = eltype(y)
    i = thread_global_index()
    offset = total_threads()
    @inbounds while i <= ndofs
        yi = zero(T)
        r = dof_cells_offset[i] : dof_cells_offset[i+1]-1
        for ind in r
            k, m = dof_cells[ind]
            yi += fes[k][m]
        end
        y[i] = yi
        i += offset
    end
    return
end
