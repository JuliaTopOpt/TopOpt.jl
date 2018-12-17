struct PrimalData{T, TV1<:AbstractVector{T}, TV2, TM<:AbstractMatrix{T}}
    σ::TV1
    α::TV1 # Lower move limit
    β::TV1 # Upper move limit
    p0::TV1
    q0::TV1
    p::TM
    q::TM
    ρ::TV2
    r::TV2
    r0::Base.RefValue{T}
    x::TV1 # Optimal value for x in dual iteration
    x1::TV1 # Optimal value for x in previous outer iteration
    f_val::Base.RefValue{T} # Function value at current iteration
    g_val::TV2 # Inequality values at current iteration
    ∇f::TV1 # Function gradient at current iteration
    ∇g::TM # Inequality gradients [var, ineq] at current iteration
end

struct XUpdater{TPD<:PrimalData}
    pd::TPD
end
function (xu::XUpdater)(λ)
    @unpack x = xu.pd
    #@show sum(x)
    #@show mean(xu.pd.p0), mean(xu.pd.q0)
    map!((j)->xu(λ, j), x, 1:length(x))
end
function (xu::XUpdater)(λ, j)
    @unpack p0, p, q0, ρ, q, σ, x1, α, β = xu.pd
    lj1 = p0[j] + matdot(λ, p, j) + dot(λ, ρ)*σ[j]/4
    lj2 = q0[j] + matdot(λ, q, j) + dot(λ, ρ)*σ[j]/4

    αj, βj = α[j], β[j]
    Lj, Uj = minus_plus(x1[j], σ[j])

    Ujαj = Uj - αj
    αjLj = αj - Lj
    ljαj = lj1/Ujαj^2 - lj2/αjLj^2 

    Ujβj = Uj - βj
    βjLj = βj - Lj
    ljβj = lj1/Ujβj^2 - lj2/βjLj^2 

    (lj1 < 0 || lj2 < 0) && @show lj1, lj2
    fpj = sqrt(lj1)
    fqj = sqrt(lj2)
    xj = (fpj * Lj + fqj * Uj) / (fpj + fqj)
    xj = ifelse(ljαj >= 0, αj, ifelse(ljβj <= 0, βj, xj))

    return xj
end

# Primal problem functions
struct ConvexApproxGradUpdater{T, TPD<:PrimalData{T}, TM<:MMAModel{T}}
    pd::TPD
    m::TM
end

function (gu::ConvexApproxGradUpdater{T})() where T
    @unpack pd, m = gu
    @unpack f_val, g_val, r = pd
    n = dim(m)
    r0 = f_val[] - mapreduce(gu, +, 1:n, init=T(0))
    map!((i)->(g_val[i] - mapreduce(gu, +, Base.Iterators.product(1:n, i:i), init=T(0))), 
        r, 1:length(constraints(m)))
    pd.r0[] = r0
end
function (gu::ConvexApproxGradUpdater{T})(j::Int) where T
    pd = gu.pd
    @unpack x, σ, x1, p0, q0, ∇f = pd
    xj = x[j]
    σj = σ[j]
    Lj, Uj = minus_plus(xj, σj) # x == x1
    ∇fj = ∇f[j]
    abs2σj∇fj = abs2(σj)*∇fj
    (p0j, q0j) = ifelse(∇fj > 0, (abs2σj∇fj, zero(T)), (zero(T), -abs2σj∇fj))
    p0[j], q0[j] = p0j, q0j
    return (p0[j] + q0[j])/σj
end
function (gu::ConvexApproxGradUpdater{T})(ji::Tuple) where T
    j, i = ji
    pd = gu.pd
    @unpack x, σ, p, q, ρ, ∇g = pd
    σj = σ[j]
    xj = x[j]
    Lj, Uj = minus_plus(xj, σj) # x == x1
    ∇gj = ∇g[j,i]
    abs2σj∇gj = abs2(σj)*∇gj
    (pji, qji) = ifelse(∇gj > 0, (abs2σj∇gj, zero(T)), (zero(T), -abs2σj∇gj))
    p[j,i], q[j,i] = pji, qji
    Δ = ρ[i]*σj/4
    return (pji + qji + 2Δ)/σj
end

struct VariableBoundsUpdater{T, TPD<:PrimalData{T}, TModel<:MMAModel{T}}
    pd::TPD
    m::TModel
    μ::T
end

function (bu::VariableBoundsUpdater{T})() where {T}
    @unpack pd, m = bu
    @unpack α, β = pd
    n = dim(m)
    s = StructArray{NTuple{2,T}}(α, β)
    map!(s, 1:n) do i
        t = bu(i)
        (x1 = t[1], x2 = t[2])
    end
end
function (bu::VariableBoundsUpdater{T})(j) where T
    @unpack m, pd, μ = bu
    @unpack x, σ = pd
    xj = x[j]
    Lj, Uj = minus_plus(xj, σ[j]) # x == x1 here
    αj = max(Lj + μ * (xj - Lj), min(m, j))
    βj = min(Uj - μ * (Uj - xj), max(m, j))
    return (αj, βj)
end

struct AsymptotesUpdater{T, TV<:AbstractVector{T}, TModel<:MMAModel{T}}
    m::TModel
    σ::TV
    x::TV
    x1::TV
    x2::TV
    s_init::T
    s_incr::T
    s_decr::T
end

struct InitialAsymptotesUpdater{T, TModel<:MMAModel{T}}
    m::TModel
    s_init::T
end
Initial(au::AsymptotesUpdater) = InitialAsymptotesUpdater(au.m, au.s_init)

function (au::AsymptotesUpdater{T, TV})(k::Iteration) where {T, TV}
    @unpack σ, m = au
    if k.i == 1 || k.i == 2
        map!(Initial(au), σ, 1:dim(m))
    else
        map!(au, σ, 1:dim(m))
    end
end
# Update move limits
function (au::InitialAsymptotesUpdater)(j::Int)
    @unpack m, s_init = au
    return s_init * (max(m, j) - min(m, j))
end
function (au::AsymptotesUpdater{T})(j::Int) where T
    @unpack x, x1, x2, s_incr, s_decr, σ, m = au
    σj = σ[j]
    xj = x[j]
    x1j = x1[j]
    x2j = x2[j]
    d = ifelse((xj == x1j || x1j == x2j), 
        σj, ifelse(xor(xj > x1j, x1j > x2j), 
        σj * s_decr, σj * s_incr))
    diff = max(m, j) - min(m, j)
    _min = T(0.01)*diff
    _max = 10diff
    return ifelse(d <= _min, _min, ifelse(d >= _max, _max, d))
end
