const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

macro thread_local_index()
    :((threadIdx().z - 1) * blockDim().y * blockDim().x + (threadIdx().y - 1) * blockDim().x + threadIdx().x)
end

macro total_threads_per_block()
    :(blockDim().z * blockDim().y * blockDim().x)
end

macro block_index()
    :(blockIdx().x + (blockIdx().y - 1) * gridDim().x + (blockIdx().z - 1) * gridDim().x * gridDim().y)
end

macro total_blocks()
    :(gridDim().z * gridDim().x * gridDim().y)
end

macro thread_global_index()
    :((@block_index() - 1) * (blockDim().x * blockDim().y * blockDim().z) + @thread_local_index())
end

macro total_threads()
    :(@total_blocks() * @total_threads_per_block())
end

function callkernel(dev, kernel, args)
    blocks, threads = getvalidconfig(dev, kernel, args)
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
