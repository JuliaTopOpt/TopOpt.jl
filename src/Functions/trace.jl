abstract type AbstractMeanMethod end
abstract type AbstractExactMeanMethod <: AbstractMeanMethod end
abstract type AbstractTraceEstimationMeanMethod <: AbstractMeanMethod end

struct ExactMean{TF} <: AbstractExactMeanMethod
    F::TF
end
ExactMean() = ExactMean(nothing)

struct ExactSVDMean{TUS} <: AbstractExactMeanMethod
    US::TUS
    n::Int
end
ExactSVDMean() = ExactSVDMean(nothing, 0)
function ExactSVDMean(F::SparseMatrixCSC)
    rows = unique(F.rowval)
    Fc = Matrix(F[rows, :])
    svdfact = svd(Fc)
    threshold = 1e-6
    @show length(svdfact.S)
    inds = findall(x -> x > threshold, svdfact.S)
    @show length(inds)
    US_dense = svdfact.U[:, inds] * Diagonal(svdfact.S[inds])
    I = Int[]
    J = Int[]
    V = eltype(F)[]
    for j = 1:length(inds)
        for i = 1:length(rows)
            push!(I, rows[i])
            push!(J, inds[j])
            push!(V, US_dense[i, j])
        end
    end
    US = sparse(I, J, V, size(F, 1), length(inds))
    return ExactSVDMean(US, size(F, 2))
end

struct TraceEstimationMean{TF,TV,TM} <: AbstractTraceEstimationMeanMethod
    F::TF
    V::TV
    sample_once::Bool
    sample_method::TM
end
function TraceEstimationMean(
    F::SparseMatrixCSC,
    nv::Int,
    sample_once::Bool = true,
    sample_method = hutch_rand!,
)
    V = zeros(eltype(F), size(F, 2), nv)
    sample_method(V)
    return TraceEstimationMean(F, V, sample_once, sample_method)
end
struct TraceEstimationSVDMean{TUS,TV,TM} <: AbstractTraceEstimationMeanMethod
    US::TUS
    n::Int
    V::TV
    sample_once::Bool
    sample_method::TM
end
function TraceEstimationSVDMean(
    F::SparseMatrixCSC,
    nv::Int,
    sample_once::Bool = true,
    sample_method = hutch_rand!,
)
    US = ExactSVDMean(F).US
    V = zeros(eltype(F), size(US, 2), nv)
    sample_method(V)
    sample_once = sample_method === hadamard! || sample_once
    return TraceEstimationSVDMean(US, size(F, 2), V, sample_once, sample_method)
end
function TraceEstimationSVDMean(
    F::SparseMatrixCSC,
    V::AbstractMatrix,
    sample_once::Bool = true,
    sample_method = hutch_rand!,
)
    US = ExactSVDMean(F).US
    sample_once = sample_method === hadamard! || sample_once
    return TraceEstimationSVDMean(US, size(F, 2), V, sample_once, sample_method)
end

abstract type AbstractDiagonalMethod end
abstract type AbstractExactDiagonalMethod <: AbstractDiagonalMethod end
abstract type AbstractDiagonalEstimationMethod <: AbstractDiagonalMethod end

struct ExactDiagonal{TF,TY,Ttemp} <: AbstractExactDiagonalMethod
    F::TF
    Y::TY # K^-1 F
    temp::Ttemp # f' K^-1 dK/d(filtered x_e) K^-1 f for all e and any one f
end
function ExactDiagonal(F::SparseMatrixCSC, nE::Int)
    return ExactDiagonal(F, zeros(eltype(F), size(F)...), zeros(eltype(F), nE))
end

struct ExactSVDDiagonal{TF,TUS,TV,TQ,TY,Ttemp} <: AbstractExactDiagonalMethod
    F::TF
    US::TUS
    V::TV
    Q::TQ # K^-1 US
    Y::TY # Q' dK/d(filtered x_e) Q without the filter
    temp::Ttemp
end
function ExactSVDDiagonal(F::SparseMatrixCSC, nE::Int)
    rows = unique(F.rowval)
    Fc = Matrix(F[rows, :])
    svdfact = svd(Fc)
    threshold = 1e-6
    @show length(svdfact.S)
    inds = findall(x -> x > threshold, svdfact.S)
    @show length(inds)
    US_dense = svdfact.U[:, inds] * Diagonal(svdfact.S[inds])
    V = svdfact.V[:, inds]
    I = Int[]
    J = Int[]
    vals = eltype(F)[]
    for j = 1:length(inds)
        for i = 1:length(rows)
            push!(I, rows[i])
            push!(J, inds[j])
            push!(vals, US_dense[i, j])
        end
    end
    US = sparse(I, J, vals, size(F, 1), length(inds))
    Q = zeros(eltype(F), size(US))
    Y = zeros(eltype(F), length(inds), length(inds))
    temp = zeros(eltype(F), nE)
    return ExactSVDDiagonal(F, US, V, Q, Y, temp)
end

struct DiagonalEstimation{TF,TY,TQ,TV,Ttemp,TM} <: AbstractDiagonalEstimationMethod
    F::TF
    V::TV # all v_i
    Y::TY # K^-1 F v_i for all i
    Q::TQ # K^-1 F * (w .* v_i) for all i
    temp::Ttemp # (w .* v_i)' F' K^-1 dK/d(filtered x_e) K^-1 F v_i for all e and any one v_i
    sample_once::Bool
    sample_method::TM
end
function DiagonalEstimation(
    F::SparseMatrixCSC,
    nv::Int,
    nE::Int,
    sample_once::Bool = true,
    sample_method = hadamard!,
)
    V = zeros(eltype(F), size(F, 2), nv)
    sample_method(V)
    Y = zeros(eltype(F), size(F, 1), nv)
    Q = similar(Y)
    temp = zeros(eltype(F), nE)
    sample_once = sample_method === hadamard! || sample_once
    return DiagonalEstimation(F, V, Y, Q, temp, sample_once, sample_method)
end
function DiagonalEstimation(
    F::SparseMatrixCSC,
    V::AbstractMatrix,
    nE::Int,
    sample_once::Bool = true,
    sample_method = hadamard!,
)
    nv = size(V, 2)
    Y = zeros(eltype(F), size(F, 1), nv)
    Q = similar(Y)
    temp = zeros(eltype(F), nE)
    sample_once = sample_method === hadamard! || sample_once
    return DiagonalEstimation(F, V, Y, Q, temp, sample_once, sample_method)
end

for T in (
    :ExactMean,
    :ExactSVDMean,
    :TraceEstimationMean,
    :TraceEstimationSVDMean,
    :ExactDiagonal,
    :ExactSVDDiagonal,
    :DiagonalEstimation,
)
    @eval baretype(::$T) = $T
end
