const StaticMatrices{m,T} = Union{StaticMatrix{m,m,T}, Symmetric{T,<:StaticMatrix{m,m,T}}}

function matrix_free_apply2Kes!(elementinfo::ElementFEAInfo{dim, T}, raw_elementinfo::ElementFEAInfo, problem::StiffnessTopOptProblem) where {dim, T}
    KK = elementinfo.Kes
    raw_KK = raw_elementinfo.Kes

    ch = problem.ch
    dof_cells = problem.metadata.dof_cells
    dof_cells_offset = problem.metadata.dof_cells_offset

    M = zero(T)
    for i in 1:length(raw_KK)
        s = sumdiag(raw_KK[i])
        #M += s * ifelse(black[i], 1, ifelse(white[i], xmin, density(vars[varind[i]], xmin))
        M += s
    end
    M /= length(dof_cells_offset)

    m = size(KK[1], 1)
    for ind in 1:length(ch.values)
        d = ch.prescribed_dofs[ind]
        v = ch.values[ind]
        r = dof_cells_offset[d] : dof_cells_offset[d+1]-1
        for idx in r
            (i,j) = dof_cells[idx]
            for col in 1:m
                for row in 1:m
                    if row == j || col == j
                        if row == col
                            if eltype(KK) <: Symmetric
                                KK[i].data[j,j] = M                            
                            else
                                KK[i][j,j] = M
                            end
                        else
                            if eltype(KK) <: Symmetric
                                KK[i].data[row,col] = zero(T)
                            else
                                KK[i][row,col] = zero(T)
                            end
                        end
                    end
                end
            end
        end
    end
    return M
end

function matrix_free_apply2f!(f::Vector{T}, rawelementinfo::ElementFEAInfo{dim, T}, M, vars, problem::StiffnessTopOptProblem, penalty, xmin, applyzero::Bool=false) where {dim, T}
    raw_KK = rawelementinfo.Kes

    ch = problem.ch
    black = problem.black
    white = problem.white
    varind = problem.varind
    dof_cells = problem.metadata.dof_cells
    dof_cells_offset = problem.metadata.dof_cells_offset
    cell_dofs = problem.metadata.cell_dofs

    m = size(raw_KK[1], 1)
    for ind in 1:length(ch.values)
        d = ch.prescribed_dofs[ind]
        v = ch.values[ind]

        r = dof_cells_offset[d] : dof_cells_offset[d+1]-1
        if !applyzero && v != 0 
            for idx in r
                (i,j) = dof_cells[idx]
                if black[i]
                    for row in 1:m
                        f[cell_dofs[row,i]] -= v * raw_KK[i][row,j]
                    end
                elseif white[i]
                    px = penalty(xmin)
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                else
                    px = penalty(density(vars[varind[i]], xmin))
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                end
            end
        end
        f[d] = zero(T)
        for idx in r 
            (i,j) = dof_cells[idx]
            if black[i]
                for col in 1:m
                    for row in 1:m
                        if row == j || col == j
                            if row == col
                                f[cell_dofs[j,i]] += M*v
                            end
                        end
                    end
                end
            elseif white[i]
                px = penalty(xmin)
                for col in 1:m
                    for row in 1:m
                        if row == j || col == j
                            if row == col
                                f[cell_dofs[j,i]] += px*M*v #Take out and do most of the rest once
                            end
                        end
                    end
                end
            else
                px = penalty(density(vars[varind[i]], xmin))
                for col in 1:m
                    for row in 1:m
                        if row == j || col == j
                            if row == col
                                f[cell_dofs[j,i]] += px*M*v
                            end
                        end
                    end
                end
            end
        end
    end
    return 
end

@generated function sumdiag(K::StaticMatrices{m,T}) where {m,T}
    return reduce((ex1,ex2) -> :($ex1 + $ex2), [:(K[$j,$j]) for j in 1:m])
end
