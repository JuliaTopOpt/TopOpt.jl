struct DualSolution{dev, T, TVCPU <: AbstractVector{T}, TVGPU}
    cpu::TVCPU
    gpu::TVGPU
end

DualSolution(x) = DualSolution{:CPU}(x)
DualSolution{:CPU}(x) = DualSolution{:CPU, eltype(x), typeof(x), Nothing}(x, nothing)
Base.getindex(d::DualSolution, i...) = d.cpu[i...]
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
    φ0 = r0 + dot(λ, r)
    φ = computeφ(φ0, p0, q0, p, q, ρ, σ, dobj.λ, x1, x)
    return -φ
end
function computeφ(φ0, p0::AbstractVector{T}, q0, p, q, ρ, σ, λ, x1, x) where {T}
    nv, nc = size(p)
    φ = φ0
    φ += tmapreduce(+, 1:nv, init = zero(T)) do j
        getdualterm(p0, q0, σ, x1, x, j)
    end
    for i in 1:nc
        λi = λ.cpu[i]
        ρi = ρ[i]
        φ += tmapreduce(+, 1:nv, init = zero(T)) do j
            getdualterm(p, q, ρi, σ, x1, x, λi, (j, i))
        end
    end
    return φ
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
    T = eltype(λ)
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
        ∇φ[i] -= tmapreduce(+, 1:nv, init = zero(T)) do j
            getgradterm(x, x1, p, q, ρi, σ, (j, i))
        end
    end
    return ∇φ
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
