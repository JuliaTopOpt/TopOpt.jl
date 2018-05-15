mutable struct MatrixFreeOperator{T, dim, TS<:StiffnessTopOptProblem{dim, T}, TK<:AbstractMatrix{T}, Tf<:AbstractVector{T}, TKes<:AbstractVector{TK}, Tfes<:AbstractVector{Tf}, Tcload<:AbstractVector{T}, TP, refshape, TCV<:CellValues{dim, T, refshape}, dimless1, TFV<:FaceValues{dimless1, T, refshape}}
    elementinfo::ElementFEAInfo{dim, T, TK, Tf, TKes, Tfes, Tcload, refshape, TCV, dimless1, TFV}
    meandiag::T
    problem::TS
    vars::Vector{T}
    xmin::T
    penalty::TP
end

import Base: *, A_mul_B!
#const nthreads = Threads.nthreads()
function A_mul_B!(y, A::MatrixFreeOperator, x)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    #division = ceil(Int, nels / nthreads)
    dofspercell = size(A.elementinfo.Kes[1], 1)
    
    Kes = A.elementinfo.Kes
    fes = A.elementinfo.fes # Used as buffers

    metadata = A.problem.metadata
    cell_dofs = metadata.cell_dofs
    dof_cells = metadata.dof_cells
    dof_cells_offset = metadata.dof_cells_offset
    black = A.problem.black
    white = A.problem.white
    penalty = A.penalty
    xmin = A.xmin
    vars = A.vars
    varind = A.problem.varind

    for i in 1:nels
        if black[i]
            px = one(T)
        elseif white[i]
            px = penalty(xmin)
        else
            px = penalty(density(vars[varind[i]], xmin))
        end
        for j in 1:dofspercell
            fes[i][j] = x[cell_dofs[j,i]]
        end
        fes[i] = px * Kes[i] * fes[i]
    end

    y .= zero(eltype(y))
    for i in 1:ndofs
        r = dof_cells_offset[i] : dof_cells_offset[i+1]-1
        for ind in r
            k, m = dof_cells[ind]
            y[i] += fes[k][m]
        end
    end
    y
end
function *(A::MatrixFreeOperator, x)
    y = zeros(x)
    A_mul_B!(y, A::MatrixFreeOperator, x)
    y
end
