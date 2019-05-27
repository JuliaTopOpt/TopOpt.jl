using ..CUDASupport
using ..TopOpt: @init_cuda
@init_cuda()
using ..GPUUtils
import ..TopOpt: whichdevice, CPU, GPU

### Utilities

MatrixOf(::Type{CuVector{T}}) where T = CuMatrix{T}

### Model

whichdevice(m::Model) = whichdevice(m.box_max)
eval_objective(::GPU, m, x::GPUVector{T}, ∇g) where {T} = T(m.objective(x, ∇g))
function eval_objective(::GPU, m, x::AbstractVector{T}, ∇g) where {T}
    x_gpu = CuArray(x)
    ∇g_gpu = CuArray(∇g)
    obj = T(m.objective(x_gpu, ∇g_gpu))
    copyto!(∇g, ∇g_gpu)
    return obj
end
function eval_objective(::CPU, m, x::CuVector{T}, ∇g) where {T}
    error("Optimization on the GPU with the objective evaluation on the CPU is weird!")
end

eval_constraint(::GPU, m, i, x::GPUVector{T}, ∇g) where T = T(constraint(m, i)(x, ∇g))
function eval_constraint(::GPU, m, i, x::AbstractVector{T}, ∇g) where T
    x_gpu = CuArray(x)
    ∇g_gpu = CuArray(∇g)
    constr = T(constraint(m, i)(x_gpu, ∇g_gpu))
    copyto!(∇g, ∇g_gpu)
    return constr
end
function eval_constraint(::CPU, m, i, x::GPUVector, ∇g)
    error("Optimization on the GPU with the constraint evaluation on the CPU is weird!")
end

Model(::GPU, args...; kwargs...) = Model{Float64, CuVector{Float64}, Vector{Function}}(args...; kwargs...)

### Dual

whichdevice(::DualSolution{:CPU}) = CPU()
whichdevice(::DualSolution{:GPU}) = GPU()
function DualSolution{:GPU}(x)
    cux = CuArray(x)
    DualSolution{:GPU, eltype(x), typeof(x), typeof(cux)}(x, cux)
end
togpu!(d::DualSolution{:GPU}) = copyto!(d.gpu, d.cpu)
tocpu!(d::DualSolution{:GPU}) = copyto!(d.cpu, d.gpu)

function computeφ(φ0, p0::CuVector{T}, q0, p, q, ρ, σ, λ, x1, x, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    nv, nc = size(p)

    φ = φ0
    result = similar(x, T, (blocksize,))
    args = (x, x1, p0, q0, σ, result, Val(threads))
    @cuda blocks = blocksize threads = threads computeφ_kernel1(args...)
    CUDAnative.synchronize()
    φ += sum(Array(result))

    for i in 1:nc
        args = (x, x1, p, q, ρ[i], σ, λ.cpu[i], i, result, Val(threads))
        @cuda blocks = blocksize threads = threads computeφ_kernel2(args...)
        CUDAnative.synchronize()
        φ += sum(Array(result))
    end

    return φ
end

function computeφ_kernel1(x::AbstractVector{T}, x1, p0, q0, σ, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        getdualterm(p0, q0, σ, x1, x, j)
    end)

    return
end

function computeφ_kernel2(x::AbstractVector{T}, x1, p, q, ρi, σ, λi, i, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        getdualterm(p, q, ρi, σ, x1, x, λi, (j, i))
    end)

    return
end

function compute_grad!(::GPU, dgrad, ∇φ::AbstractVector{T}, λ, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    @unpack pd = dgrad
    @unpack r, r0, p, q, σ, x1, x, ρ = pd
    # Updates x to the Lagrangian minimizer for the input λ
    dgrad.λ.cpu .= λ
    dgrad.x_updater(dgrad.λ)
    nv, nc = size(p)
    # Negate since we have a maximization problem
    for i in 1:nc
        ∇φ[i] = -r[i]
        result = similar(x, T, (blocksize,))
        args = (x, x1, p, q, ρ[i], i, σ, result, Val(threads))
        @cuda blocks = blocksize threads = threads compute_grad_kernel(args...)
        CUDAnative.synchronize()
        ∇φ[i] -= sum(Array(result))
    end
    return ∇φ
end
function compute_grad_kernel(x::AbstractVector{T}, x1, p, q, ρi, i, σ, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        getgradterm(x, x1, p, q, ρi, σ, (j, i))
    end)

    return
end

### Primal

whichdevice(p::PrimalData) = whichdevice(p.x)

function (xu::XUpdater{T, TV})(λ::DualSolution{:GPU}) where {T, TV <: CuVector{T}}
    @unpack x, p0, p, q0, q, σ, x1, α, β, ρ = xu.pd
    λρ = dot(λ.cpu, ρ)
    togpu!(λ)
    λgpu = λ.gpu
    args = (x, λgpu, λρ, p, q, p0, q0, σ, x1, α, β)
    callkernel(dev, xupdater_kernel, args)
    CUDAdrv.synchronize(ctx)
    return
end
function xupdater_kernel(x, λgpu, λρ, p, q, p0, q0, σ, x1, α, β)
    j = @thread_global_index()
    offset = @total_threads()
    while j <= length(x)
        λpj = @matdot(λgpu, p, j)
        λqj = @matdot(λgpu, q, j)
        x[j] = getxj(λρ, λpj, λqj, j, p0, q0, σ, x1, α, β)
        j += offset
    end
    return
end

function (gu::ConvexApproxGradUpdater{T, TV})() where {T, TV <: CuVector}
    @unpack pd, m = gu
    @unpack f_x, g, r, x, σ, x1, p0, q0, ∇f_x, p, q, ρ, ∇g = pd
    n = dim(m)
    r0 = compute_r0(f_x, x, σ, x1, p0, q0, ∇f_x)
    update_r!(r, g, x, σ, p, q, ρ, ∇g)
    pd.r0 = r0
end

function compute_r0(r0, x::AbstractVector{T}, σ, x1, p0, q0, ∇f_x, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    result = similar(x, T, (blocksize,))
    args = (x, σ, x1, p0, q0, ∇f_x, result, Val(threads))
    @cuda blocks = blocksize threads = threads gradupdater_kernel1(args...)
    CUDAnative.synchronize()
    r0 -= sum(Array(result))
    return r0
end

function gradupdater_kernel1(x::AbstractVector{T}, σ, x1, p0, q0, ∇f_x, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        getgradelement(x, σ, x1, p0, q0, ∇f_x, j)
    end)

    return
end

function update_r!(r, g, x::AbstractVector{T}, σ, p, q, ρ, ∇g, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    result = similar(x, T, (blocksize,))
    for i in 1:length(r)
        r[i] = g[i]
        args = (x, σ, p, q, i, ρ[i], ∇g, result, Val(threads))
        @cuda blocks = blocksize threads = threads gradupdater_kernel2(args...)
        CUDAnative.synchronize()
        r[i] -= sum(Array(result))
    end
    return 
end

function gradupdater_kernel2(x::AbstractVector{T}, σ, p, q, i, ρi, ∇g, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        getgradelement(x, σ, p, q, ρi, ∇g, (j, i))
    end)

    return
end

function (bu::VariableBoundsUpdater{T, TV})() where {T, TV <: CuVector}
    @unpack pd, m, μ = bu
    @unpack α, β, x, σ = pd
    @unpack box_max, box_min = m
    n = dim(m)
    args = (α, β, σ, x, box_max, box_min)
    callkernel(dev, bounds_kernel, args)
    CUDAdrv.synchronize(ctx)
    return
end

function bounds_kernel(α, β, σ, x, box_max, box_min)
    j = @thread_global_index()
    offset = @total_threads()
    while j <= length(σ)
        xj = x[j]
        Lj, Uj = minus_plus(xj, σ[j]) # x == x1 here
        αj = max(Lj + μ * (xj - Lj), box_min[j])
        βj = min(Uj - μ * (Uj - xj), box_max[j])
        α[j] = αj
        β[j] = βj
        j += offset
    end
    return
end

function (au::AsymptotesUpdater{T, TV})(k::Iteration) where {T, TV <: CuVector}
    @unpack σ, m, s_init, s_incr, s_decr, x, x1, x2 = au
    @unpack box_max, box_min = m

    if k.i == 1 || k.i == 2
        args = (σ, s_init, box_max, box_min)
        callkernel(dev, asymptotes_kernel1, args)
        CUDAdrv.synchronize(ctx)
    else
        args = (σ, x, x1, x2, box_max, box_min, s_incr, s_decr)
        callkernel(dev, asymptotes_kernel2, args)
        CUDAdrv.synchronize(ctx)
    end
end

function asymptotes_kernel1(σ, s_init, box_max, box_min)
    j = @thread_global_index()
    offset = @total_threads()
    while j <= length(σ)
        σ[j] = s_init * (box_max[j] - box_min[j])
        j += offset
    end
    return
end

function asymptotes_kernel2(σ::AbstractVector{T}, x, x1, x2, box_max, box_min, s_incr, s_decr) where {T}
    j = @thread_global_index()
    offset = @total_threads()
    while j <= length(σ)
        σj = σ[j]
        xj = x[j]
        x1j = x1[j]
        x2j = x2[j]
        d = ifelse((xj == x1j || x1j == x2j), 
            σj, ifelse(xor(xj > x1j, x1j > x2j), 
            σj * s_decr, σj * s_incr))
        diff = box_max[j] - box_min[j]
        _min = T(0.01)*diff
        _max = 10diff
        σ[j] = ifelse(d <= _min, _min, ifelse(d >= _max, _max, d))
        j += offset
    end
    return
end

### Lift

function computew(x::CuVector{T}, x1, σ, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    result = similar(x, T, (blocksize,))
    args = (x, x1, σ, result, Val(threads))
    @cuda blocks = blocksize threads = threads computew_kernel(args...)
    CUDAnative.synchronize()
    w = sum(Array(result))
end
function computew_kernel(x::AbstractVector{T}, x1, σ, result, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(j, length(x), +, T, LMEM, result, begin
        (x[j] - x1[j])^2 / (σ[j]^2 - (x[j] - x1[j])^2)
    end)
    
    return
end
