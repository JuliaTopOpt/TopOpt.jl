const StaticMatrices{m,T} = Union{StaticMatrix{m,m,T}, Symmetric{T,<:StaticMatrix{m,m,T}}}

function matrix_free_apply2Kes!(elementinfo::ElementFEAInfo{dim, T}, raw_elementinfo::ElementFEAInfo, problem::StiffnessTopOptProblem) where {dim, T}
    KK = elementinfo.Kes
    raw_KK = raw_elementinfo.Kes

    ch = problem.ch
    dof_cells = elementinfo.metadata.dof_cells

    M = mapreduce(sumdiag, +, raw_KK, init=zero(T))
    #M = zero(T)
    #for i in 1:length(raw_KK)
    #    s = sumdiag(raw_KK[i])
        #M += s * ifelse(black[i], 1, ifelse(white[i], xmin, density(vars[varind[i]], xmin))
    #    M += s
    #end
    M /= length(dof_cells.offsets) - 1
    m = size(KK[1], 1)
    update_KK!(KK, m, M, ch.values, ch.prescribed_dofs, dof_cells)
    return M
end

function update_KK!(KK::Vector, m, M::T, values, prescribed_dofs, dof_cells) where {T}
    for ind in 1:length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        r = dof_cells.offsets[d] : dof_cells.offsets[d+1]-1
        for idx in r
            (i,j) = dof_cells.values[idx]
            if eltype(KK) <: Symmetric
                KKi = KK[i].data
            else
                KKi = KK[i]
            end
            for col in 1:m
                for row in 1:m
                    if row == j || col == j
                        if row == col
                            KKi = @set KKi[j,j] = M
                        else
                            KKi = @set KKi[row,col] = zero(T)
                        end
                    end
                end
            end
            if eltype(KK) <: Symmetric
                KK[i] = Symmetric(KKi)
            else
                KK[i] = KKi
            end
        end
    end
    return
end

function update_KK!(KK::CuVector, m, M, values, prescribed_dofs, dof_cells)
    args = (KK, m, M, values, prescribed_dofs, dof_cells.offsets, dof_cells.values)
    callkernel(dev, bc_kernel1, args)
    CUDAdrv.synchronize(ctx)
    return 
end

function bc_kernel1(KK, m, M::T, values, prescribed_dofs, dof_cells_offsets, dof_cells_values) where {T}
    ind = @thread_global_index()
    offset = @total_threads()
    while ind <= length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        r = dof_cells_offsets[d] : dof_cells_offsets[d+1]-1
        for idx in r
            (i,j) = dof_cells_values[idx]
            if eltype(KK) <: Symmetric
                KKi = KK[i].data
            else
                KKi = KK[i]
            end
            for col in 1:m
                for row in 1:m
                    if row == j || col == j
                        if row == col
                            KKi = @set KKi[j,j] = M
                        else
                            KKi = @set KKi[row,col] = zero(T)
                        end
                    end
                end
            end
            if eltype(KK) <: Symmetric
                KK[i] = Symmetric(KKi)
            else
                KK[i] = KKi
            end
        end
        ind += offset
    end
    return
end

function matrix_free_apply2f!(f::AbstractVector{T}, rawelementinfo::ElementFEAInfo{dim, T}, M, vars, problem::StiffnessTopOptProblem, penalty, xmin, applyzero::Bool=false) where {dim, T}
    raw_KK = rawelementinfo.Kes

    ch = problem.ch
    black = rawelementinfo.black
    white = rawelementinfo.white
    varind = rawelementinfo.varind
    dof_cells = rawelementinfo.metadata.dof_cells
    cell_dofs = rawelementinfo.metadata.cell_dofs

    update_f!(f, ch.values, ch.prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, raw_KK, xmin, penalty, vars, varind, M)

    return
end

function update_f!(f::Vector{T}, values, prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, raw_KK, xmin, penalty, vars, varind, M) where {T}
    for ind in 1:length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(raw_KK[ind], 1)

        r = dof_cells.offsets[d] : dof_cells.offsets[d+1]-1
        if !applyzero && v != 0 
            for idx in r
                (i,j) = dof_cells.values[idx]
                if black[i]
                    for row in 1:m
                        f[cell_dofs[row,i]] -= v * raw_KK[i][row,j]
                    end
                elseif white[i]
                    px = xmin
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                else
                    px = density(penalty(vars[varind[i]]), xmin)
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                end
            end
        end
        f[d] = zero(T)
        for idx in r 
            (i,j) = dof_cells.values[idx]
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
                px = xmin
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
                px = density(penalty(vars[varind[i]]), xmin)
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

function update_f!(f::CuVector{T}, values, prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, raw_KK, xmin, penalty, vars, varind, M) where {T}
    args = (f, values, prescribed_dofs, applyzero, dof_cells.offsets, dof_cells.values, cell_dofs, black, white, raw_KK, xmin, penalty, vars, varind, M)
    callkernel(dev, bc_kernel2, args)
    CUDAdrv.synchronize(ctx)
    return 
end

function bc_kernel2(f::AbstractVector{T}, values, prescribed_dofs, applyzero, dof_cells_offsets, dof_cells_values, cell_dofs, black, white, raw_KK, xmin, penalty, vars, varind, M) where {T}

    ind = @thread_global_index()
    offset = @total_threads()
    while ind <= length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(raw_KK[ind], 1)

        r = dof_cells_offsets[d] : dof_cells_offsets[d+1]-1
        if !applyzero && v != 0 
            for idx in r
                (i,j) = dof_cells_values[idx]
                if black[i]
                    for row in 1:m
                        f[cell_dofs[row,i]] -= v * raw_KK[i][row,j]
                    end
                elseif white[i]
                    px = xmin
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                else
                    px = density(penalty(vars[varind[i]]), xmin)
                    for row in 1:m
                        f[cell_dofs[row,i]] -= px * v * raw_KK[i][row,j]
                    end
                end
            end
        end
        f[d] = zero(T)
        for idx in r 
            (i,j) = dof_cells_values[idx]
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
                px = xmin
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
                px = density(penalty(vars[varind[i]]), xmin)
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
        ind += offset
    end
    return 
end

@generated function sumdiag(K::StaticMatrices{m,T}) where {m,T}
    return reduce((ex1,ex2) -> :($ex1 + $ex2), [:(K[$j,$j]) for j in 1:m])
end
