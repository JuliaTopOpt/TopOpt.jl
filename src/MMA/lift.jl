struct LiftUpdater{T, TV<:AbstractVector{T}, TPD}
    pd::TPD
    ρ::TV
    g::TV
    ng_approx::TV
    n_j::Int
end

function (lu::LiftUpdater{T})() where T
    @unpack ρ, pd = lu
    @unpack x, x1, σ  = pd
    n_i = length(ρ)
    w = computew(x, x1, σ)
    w /= 2
    lift = false
    for i in 1:n_i
        lift = lift || lu(w, i)
    end
    return lift
end
function computew(x::Vector{T}, x1, σ) where {T}
    w = zero(T)
    for j in 1:length(x)
        w += (x[j] - x1[j])^2 / (σ[j]^2 - (x[j] - x1[j])^2)
    end
    w
end

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

function (lu::LiftUpdater{T})(w, i::Int) where T
    @unpack ρ, g, ng_approx = lu
    gi, g_approxi = g[i], -ng_approx[i]
    lift = (gi > g_approxi)
    δi = (gi - g_approxi)/w
    ρi = ρ[i]
    ρ[i] = ifelse(lift, min(10ρi, T(1.1)*(ρi+δi)), ρi)
    return lift
end

struct LiftResetter{T, TV}
    ρ::TV
    ρmin::T
end
function (lr::LiftResetter)(k::Iteration)
    @unpack ρ = lr
    if k.i > 1
        map!(lr, ρ, 1:length(ρ))
    end
    return
end
function (lr::LiftResetter{T})(i::Int) where T
    @unpack ρmin, ρ = lr
    return max(T(0.1)*ρ[i], ρmin)
end
