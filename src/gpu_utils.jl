const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

thread_local_index() = (threadIdx().z - 1) * blockDim().y * blockDim().x + (threadIdx().y - 1) * blockDim().x + threadIdx().x

total_threads_per_blocks() = blockDim().z * blockDim().y * blockDim().x

block_index() = blockIdx().x + (blockIdx().y - 1) * gridDim().x + (blockIdx().z - 1) * gridDim().x * gridDim().y

total_blocks() = gridDim().z * gridDim().x * gridDim().y

thread_global_index() = (block_index() - 1) * (blockDim().x * blockDim().y * blockDim().z) + local_index()

total_threads() = total_blocks() * total_threads_per_block()
