abstract type AbstractMatrixOperator{Tconv} end

struct MatrixOperator{Tconv,TK,Tf} <: AbstractMatrixOperator{Tconv}
    K::TK
    f::Tf
    conv::Tconv
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::MatrixOperator)
    return println("TopOpt matrix linear operator")
end
LinearAlgebra.mul!(c, op::MatrixOperator, b) = mul!(c, op.K, b)
Base.size(op::MatrixOperator, i...) = size(op.K, i...)
Base.eltype(op::MatrixOperator) = eltype(op.K)
LinearAlgebra.:*(op::MatrixOperator, b) = mul!(similar(b), op.K, b)

struct MatrixFreeOperator{
    Tconv,
    T,
    dim,
    Tf<:AbstractVector{T},
    Te<:ElementFEAInfo{dim,T},
    Tv<:AbstractVector{T},
    Tx,
    Tf1,
    Tf2,
    Tp,
} <: AbstractMatrixOperator{Tconv}
    f::Tf
    elementinfo::Te
    meandiag::T
    vars::Tv
    xes::Tx
    fixed_dofs::Tf1
    free_dofs::Tf2
    xmin::T
    penalty::Tp
    conv::Tconv
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::MatrixFreeOperator)
    return println("TopOpt matrix-free linear operator")
end
Base.size(op::MatrixFreeOperator) = (size(op, 1), size(op, 2))
Base.size(op::MatrixFreeOperator, i) = 1 <= i <= 2 ? length(op.elementinfo.fixedload) : 1
Base.eltype(op::MatrixFreeOperator{<:Any,T}) where {T} = T

import LinearAlgebra: *, mul!

function *(A::MatrixFreeOperator, x)
    y = similar(x)
    mul!(y, A::MatrixFreeOperator, x)
    return y
end

function mul!(y::TV, A::MatrixFreeOperator, x::TV) where {TV<:AbstractVector}
    T = eltype(y)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    dofspercell = size(A.elementinfo.Kes[1], 1)
    meandiag = A.meandiag

    @unpack Kes, metadata, black, white, varind = A.elementinfo
    @unpack cell_dofs, dof_cells = metadata
    @unpack penalty, xmin, vars, fixed_dofs, free_dofs, xes = A

    for i in 1:nels
        if PENALTY_BEFORE_INTERPOLATION
            px = ifelse(
                black[i],
                one(T),
                ifelse(white[i], xmin, density(penalty(vars[varind[i]]), xmin)),
            )
        else
            px = ifelse(
                black[i],
                one(T),
                ifelse(white[i], xmin, penalty(density(vars[varind[i]], xmin))),
            )
        end
        xe = xes[i]
        for j in 1:dofspercell
            xe = @set xe[j] = x[cell_dofs[j, i]]
        end
        if eltype(Kes) <: Symmetric
            xes[i] = px * (bcmatrix(Kes[i]).data * xe)
        else
            xes[i] = px * (bcmatrix(Kes[i]) * xe)
        end
    end

    for i in 1:length(fixed_dofs)
        dof = fixed_dofs[i]
        y[dof] = meandiag * x[dof]
    end
    for i in 1:length(free_dofs)
        dof = free_dofs[i]
        yi = zero(T)
        r = dof_cells.offsets[dof]:(dof_cells.offsets[dof + 1] - 1)
        for ind in r
            k, m = dof_cells.values[ind]
            yi += xes[k][m]
        end
        y[dof] = yi
    end
    return y
end
