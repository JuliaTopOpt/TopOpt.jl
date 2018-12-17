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
    w = mapreduce((x_x1_σ)->((x_x1_σ[1] - x_x1_σ[2])^2 / (x_x1_σ[3]^2 - (x_x1_σ[1] - x_x1_σ[2])^2)), +, zip(x, x1, σ), init=T(0)) / 2
    return mapreduce(i->lu(w, i), or, 1:n_i, init=false)
end
function (lu::LiftUpdater{T})(w, i::Int) where T
    @unpack ρ, g, ng_approx, n_j, pd = lu
    @unpack x, x1, σ = pd
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
