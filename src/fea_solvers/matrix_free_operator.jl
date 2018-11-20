mutable struct MatrixFreeOperator{T, dim, TEInfo<:ElementFEAInfo{dim, T}, TS<:StiffnessTopOptProblem{dim, T}, Tf<:AbstractVector{T}, TP}
    elementinfo::TEInfo
    meandiag::T
    problem::TS
    vars::Tf
    xmin::T
    penalty::TP
end

import LinearAlgebra: *, mul!
#const nthreads = Threads.nthreads()
function mul!(y, A::MatrixFreeOperator, x)
    T = eltype(y)
    nels = length(A.elementinfo.Kes)
    ndofs = length(A.elementinfo.fixedload)
    #division = ceil(Int, nels / nthreads)
    dofspercell = size(A.elementinfo.Kes[1], 1)
    
    Kes = A.elementinfo.Kes
    fes = A.elementinfo.fes # Used as buffers

    metadata = A.elementinfo.metadata
    cell_dofs = metadata.cell_dofs
    dof_cells = metadata.dof_cells
    dof_cells_offset = metadata.dof_cells_offset
    black = A.elementinfo.black
    white = A.elementinfo.white
    penalty = A.penalty
    xmin = A.xmin
    vars = A.vars
    varind = A.elementinfo.varind

    map!(fes, 1:nels) do i
        px = ifelse(black[i], one(T), 
                    ifelse(white[i], xmin, 
                            density(penalty(vars[varind[i]]), xmin)
                        )
                    )
        fe = fes[i] 
        for j in 1:dofspercell
            fe = @set fe[j] = x[cell_dofs[j,i]]
        end
        fes[i] = fe
        return px * Kes[i] * fe
    end

    map!(y, 1:ndofs) do i
        yi = zero(T)
        r = dof_cells_offset[i] : dof_cells_offset[i+1]-1
        for ind in r
            k, m = dof_cells[ind]
            yi += fes[k][m]
        end
        return yi
    end
    y
end
function *(A::MatrixFreeOperator, x)
    y = zeros(x)
    mul!(y, A::MatrixFreeOperator, x)
    y
end
