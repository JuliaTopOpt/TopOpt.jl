module GPUUtils

using CUDAnative, GPUArrays
import CuArrays, CUDAdrv

export  CPU,
        GPU,
        whichdevice,
        @define_cu, 
        @thread_local_index, 
        @total_threads_per_block, 
        @block_index, 
        @total_blocks, 
        @thread_global_index, 
        @total_threads, 
        callkernel

struct CPU end
struct GPU end
whichdevice(::Any) = CPU()
whichdevice(s::AbstractArray) = s isa GPUArray ? GPU() : CPU()

function getfieldnames end
function cufieldnames end

macro define_cu(T, fields...)
    _define_cu(T, fields...)
end

function _define_cu(T, fields...)
    all_fields = gensym()
    args = gensym()
    if eltype(fields) <: QuoteNode
        field_syms = Tuple(field.value for field in fields)
    elseif eltype(fields) <: Symbol
        field_syms = fields
    else
        throw("Unsupported fields.")
    end
    esc(quote
        $all_fields = Tuple(fieldnames($T))
        @eval @inline GPUUtils.getfieldnames(::Type{<:$T}) = $(Expr(:$, all_fields))
        @inline GPUUtils.cufieldnames(::Type{<:$T}) = $field_syms
        $args = Expr[]
        for fn in $all_fields
            push!($args, :(GPUUtils._cu(s, s.$fn, $(Val(fn)))))
        end
        @eval begin
            function CuArrays.cu(s::$T)
                $T($(Expr(:$, Expr(:..., args))))
            end
        end
    end)
end

function _cu(s::T, f::F, ::Val{fn}) where {T, F, fn}
    if fn ∈ GPUUtils.cufieldnames(T)
        if F <: AbstractArray
            CuArrays.CuArray(f)
        else
            CuArrays.cu(f)
        end
    else
        f
    end
end

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
    #@show blocks, threads
    @cuda blocks=blocks threads=threads kernel(args...)

    return
end

# From CuArrays.jl
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

end # module
