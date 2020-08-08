struct CG end

struct CyclicIterator
    start::Int
    n::Int
    rev::Bool
end
CyclicIterator(start, n; rev=false) = CyclicIterator(start, n, rev)
function Base.iterate(iter::CyclicIterator, i = iter.start)
    if iter.rev
        i >= 1 && return i, i - 1
        i + iter.n > iter.start && return i + iter.n, i - 1
    else
        i <= iter.n && return i, i + 1
        i - iter.n < iter.start && return i - iter.n, i + 1
    end
    return nothing
end

@params struct ApproxInverseHessian{T} <: AbstractMatrix{T}
    S::AbstractMatrix{T}
    Y::AbstractMatrix{T}
    p::AbstractVector{T}
    prev_x::AbstractVector{T}
    prev_g::AbstractVector{T}
    oldest::Base.RefValue{Int}
    activen::Base.RefValue{Int}
end
function ApproxInverseHessian(g::AbstractVector, n::Int)
    S = similar(g, length(g), n)
    Y = similar(S)
    p = similar(g, n)
    prev_x = similar(g)
    prev_g = similar(g)
    S .= 0
    Y .= 0
    p .= 0
    prev_x .= 0
    prev_g .= 0
    oldest = 1
    return ApproxInverseHessian(S, Y, p, prev_x, prev_g, Ref(oldest), Ref(0))
end
Base.size(invH::ApproxInverseHessian) = (size(invH.S, 1), size(invH.S, 1))
function update!(invH::ApproxInverseHessian, g, x)
    @unpack S, Y, p, prev_x, prev_g, oldest, activen = invH
    n = size(S, 2)
    ind = oldest[]
    new_s = x .- prev_x
    new_y = g .- prev_g
    new_p = 1 / dot(new_s, new_y)
    if isfinite(new_p)
        S[:,ind] .= new_s
        Y[:,ind] .= new_y
        p[ind] = new_p
        oldest[] = oldest[] == n ? 1 : oldest[] + 1
        activen[] = min(activen[] + 1, n)
    else
        activen[] = 0
        oldest[] = 1
    end
    prev_x .= x
    prev_g .= g
    return invH
end
function reset!(invH::ApproxInverseHessian)
    invH.activen[] = 0
    invH.oldest[] = 1
    return invH
end
function LinearAlgebra.mul!(c::AbstractVector{T}, invH::ApproxInverseHessian{T}, b::AbstractVector{T}) where {T}
    @unpack S, Y, p, prev_x, prev_g, activen = invH
    println("    activen = $(activen[])")
    n = min(activen[], size(S, 2))
    c .= b
    n == 0 && return c
    oldest = invH.oldest[]
    newest = oldest == 1 ? n : oldest - 1
    for i in CyclicIterator(newest, n, rev=true)
        @views c .-= dot(S[:,i], c) * p[i] .* Y[:,i]
    end
    for i in CyclicIterator(oldest, n, rev=false)
        @views c .-= dot(Y[:,i], c) * p[i] .* S[:,i]
    end
    @views c .+= dot(S[:,newest], b) * p[newest] .* S[:,newest]
    return c
end

struct LBFGS{TH <: Union{Nothing, ApproxInverseHessian}}
    n::Int
    invH::TH
end
LBFGS(n::Int) = LBFGS(n, nothing)

const FluxOpt = Union{Optimise.Descent, Optimise.Momentum, Optimise.Nesterov, Optimise.RMSProp, Optimise.ADAM, Optimise.AdaMax, Optimise.ADAGrad, Optimise.ADADelta, Optimise.AMSGrad, Optimise.NADAM, Optimise.Optimiser, Optimise.InvDecay, Optimise.ExpDecay, Optimise.WeightDecay}
