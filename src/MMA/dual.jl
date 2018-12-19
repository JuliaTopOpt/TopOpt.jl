struct DualSolution{dev, T, TVCPU <: AbstractVector{T}, TVGPU <: Union{Nothing, CuVector{T}}}
    cpu::TVCPU
    gpu::TVGPU
end
whichdevice(::DualSolution{:CPU}) = CPU()
whichdevice(::DualSolution{:GPU}) = GPU()

DualSolution(x) = DualSolution{:CPU}(x)
DualSolution{:CPU}(x) = DualSolution{:CPU, eltype(x), typeof(x), Nothing}(x, nothing)
function DualSolution{:GPU}(x)
    cux = CuArray(x)
    DualSolution{:GPU, eltype(x), typeof(x), typeof(cux)}(x, cux)
end
Base.getindex(d::DualSolution, i...) = d.cpu[i...]
togpu!(d::DualSolution{:GPU}) = copyto!(d.gpu, d.cpu)
tocpu!(d::DualSolution{:GPU}) = copyto!(d.cpu, d.gpu)
togpu!(d::DualSolution{:CPU}) = nothing
tocpu!(d::DualSolution{:CPU}) = nothing

function getdualterm(p, q, ρi, σ, x1, x, λi, ji::Tuple)
    j, i = ji
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    xj = x[j]
    Ujxj = Uj - xj
    xjLj = xj - Lj
    Δ = ρi*σj/4
    return λi*((p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj)
end
function getdualterm(p0, q0, σ, x1, x, j::Int)
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    return p0[j]/(Uj-x[j]) + q0[j]/(x[j]-Lj)
end
struct DualObjVal{TPD, TXUpdater, TDualSol}
    pd::TPD
    x_updater::TXUpdater
    λ::TDualSol
end
function (dobj::DualObjVal)(λ)
    @unpack pd = dobj
    @unpack r, r0, p0, q0, σ, x1, x, p, q, ρ = pd
    T = eltype(p)
    # Updates x to the Lagrangian minimizer for the input λ
    #println("Updating x")
    dobj.λ.cpu .= λ
    dobj.x_updater(dobj.λ)
    nv, nc = size(p)
    φ0 = r0[] + dot(λ, r)
    φ = computeφ(φ0, p0, q0, p, q, ρ, σ, dobj.λ, x1, x)
    return -φ
end
function computeφ(φ0, p0::AbstractVector{T}, q0, p, q, ρ, σ, λ, x1, x) where {T}
    nv, nc = size(p)
    φ = φ0
    for j in 1:nv
        φ += getdualterm(p0, q0, σ, x1, x, j)
    end
    for i in 1:nc
        λi = λ.cpu[i]
        ρi = ρ[i]
        for j in 1:nv
            φ += getdualterm(p, q, ρi, σ, x1, x, λi, (j, i))
        end
    end
    return φ
end

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
    j = @thread_global_index()
	offset = @total_threads()
    out = zero(T)
    # # Loop sequentially over chunks of input vector
    while j <= length(x)
        out += getdualterm(p0, q0, σ, x1, x, j)
        j += offset
    end

    # Perform parallel reduction
	tmp_local = @cuStaticSharedMem(T, LMEM)
    local_index = @thread_local_index()
    tmp_local[local_index] = out
    sync_threads()

    offset = @total_threads_per_block() ÷ 2
    while offset > 0
        if (local_index <= offset)
            tmp_local[local_index] += tmp_local[local_index + offset]
        end
		sync_threads()
        offset = offset ÷ 2
    end
    if local_index == 1
        result[@block_index()] = tmp_local[1]
    end

    return
end

function computeφ_kernel2(x::AbstractVector{T}, x1, p, q, ρi, σ, λi, i, result, ::Val{LMEM}) where {T, LMEM}
    j = @thread_global_index()
	offset = @total_threads()
    out = zero(T)
    # # Loop sequentially over chunks of input vector
    while j <= length(x)
        out += getdualterm(p, q, ρi, σ, x1, x, λi, (j, i))
        j += offset
    end

    # Perform parallel reduction
	tmp_local = @cuStaticSharedMem(T, LMEM)
    local_index = @thread_local_index()
    tmp_local[local_index] = out
    sync_threads()

    offset = @total_threads_per_block() ÷ 2
    while offset > 0
        if (local_index <= offset)
            tmp_local[local_index] += tmp_local[local_index + offset]
        end
		sync_threads()
        offset = offset ÷ 2
    end
    if local_index == 1
        result[@block_index()] = tmp_local[1]
    end

    return
end

struct DualObjGrad{TPD, TXUpdater, TDualSol}
    pd::TPD
    x_updater::TXUpdater
    λ::TDualSol
end
function (dgrad::DualObjGrad)(∇φ::AbstractVector{T}, λ) where {T}
    compute_grad!(whichdevice(dgrad.pd), dgrad, ∇φ, λ)
end
function compute_grad!(::CPU, dgrad, ∇φ, λ)
    @unpack pd = dgrad
    @unpack r, r0, p, q, σ, x1, x, ρ = pd
    # Updates x to the Lagrangian minimizer for the input λ
    dgrad.λ.cpu .= λ
    dgrad.x_updater(dgrad.λ)
    nv, nc = size(p)
    # Negate since we have a maximization problem
    for i in 1:nc
        ∇φ[i] = -r[i]
        ρi = ρ[i]
        for j in 1:nv
            ∇φ[i] -= getgradterm(x, x1, p, q, ρi, σ, (j, i))
        end
    end
    return ∇φ
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
    j = @thread_global_index()
	offset = @total_threads()
    out = zero(T)
    # # Loop sequentially over chunks of input vector
    @inbounds while j <= length(x)
        out += getgradterm(x, x1, p, q, ρi, σ, (j, i))
        j += offset
    end

    # Perform parallel reduction
	tmp_local = @cuStaticSharedMem(T, LMEM)
    local_index = @thread_local_index()
    tmp_local[local_index] = out
    sync_threads()

    offset = @total_threads_per_block() ÷ 2
    while offset > 0
        if (local_index <= offset)
            tmp_local[local_index] += tmp_local[local_index + offset]
        end
		sync_threads()
        offset = offset ÷ 2
    end
    if local_index == 1
        result[@block_index()] = tmp_local[1]
    end
    return
end
function getgradterm(x, x1, p, q, ρi, σ, ji::Tuple)
    j, i = ji
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    Ujxj = Uj - x[j]
    xjLj = x[j] - Lj
    Δ = ρi*σj/4
    return (p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj
end
