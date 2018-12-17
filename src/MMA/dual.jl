struct DualTermEvaluator{T, TPD<:PrimalData{T}}
    pd::TPD
end
function (dte::DualTermEvaluator)(λ, ji::Tuple)
    j, i = ji
    @unpack pd = dte
    @unpack p, q, ρ, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    xj = x[j]
    Ujxj = Uj - xj
    xjLj = xj - Lj
    Δ = ρ[i]*σj/4
    return λ[i]*((p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj)
end
function (dte::DualTermEvaluator)(j::Int)
    pd = dte.pd
    @unpack p0, q0, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    return p0[j]/(Uj-x[j]) + q0[j]/(x[j]-Lj)
end
struct DualObjVal{T, TPD<:PrimalData}
    pd::TPD
    dte::DualTermEvaluator{T, TPD}
    x_updater::XUpdater{TPD}
end
DualObjVal(pd::TPD, x_updater::XUpdater{TPD}) where {TPD <: PrimalData} = DualObjVal(pd, DualTermEvaluator(pd), x_updater)
function (dobj::DualObjVal{T, TPD})(λ) where {T, TPD<:PrimalData{T}}
    @unpack pd, dte = dobj
    @unpack p, r, r0 = pd
    # Updates x to the Lagrangian minimizer for the input λ
    #println("Updating x")
    dobj.x_updater(λ)
    nv, nc = size(p)
    φ = r0[] + dot(λ, r)
    all(isfinite, λ) || (@show λ; error("Wait a minute, you screwed up. 0"))

    isfinite(φ) || error("Wait a minute, you screwed up. 1")
    #println("Adding objective terms")
    φ += mapreduce(dte, +, 1:nv, init=T(0))
    isfinite(φ) || error("Wait a minute, you screwed up. 2")
    #println("Adding constraint terms")
    φ += mapreduce((ji)->dte(λ, ji), +, Base.Iterators.product(1:nv, 1:nc), init=T(0))
    #@show φ, λ
    isfinite(φ) || error("Wait a minute, you screwed up. 3")
    return -φ
end

struct DualGradTermEvaluator{TPD<:PrimalData}
    pd::TPD
end
function (gte::DualGradTermEvaluator)(ji::Tuple)
    j, i = ji
    pd = gte.pd
    @unpack p, q, ρ, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    Ujxj = Uj - x[j]
    xjLj = x[j] - Lj
    Δ = ρ[i]*σj/4
    return (p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj
end
struct DualObjGrad{TPD<:PrimalData}
    pd::TPD
    gte::DualGradTermEvaluator{TPD}
    x_updater::XUpdater{TPD}
end
DualObjGrad(pd::PrimalData, x_updater::XUpdater) = DualObjGrad(pd, DualGradTermEvaluator(pd), x_updater)
function (dgrad::DualObjGrad{TPD})(∇φ::AbstractVector{T}, λ) where {T, TPD<:PrimalData{T}}
    @unpack pd, gte = dgrad
    @unpack p, r, r0 = pd
    # Updates x to the Lagrangian minimizer for the input λ
    dgrad.x_updater(λ)
    nv, nc = size(p)
    # Negate since we have a maximization problem
    map!((i)->(-r[i] - mapreduce(gte, +, Base.Iterators.product(1:nv, i:i), init=T(0))),
        ∇φ, 1:nc)
    return ∇φ
end
