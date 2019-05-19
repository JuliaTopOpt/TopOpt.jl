function matrix_free_apply2f!(f::AbstractVector{T}, elementinfo::ElementFEAInfo{dim, T}, M, vars, problem::StiffnessTopOptProblem, penalty, xmin, applyzero::Bool=false) where {dim, T}
    @unpack Kes, black, white, varind, metadata = elementinfo
    @unpack dof_cells, cell_dofs = metadata
    @unpack ch = problem
    @unpack values, prescribed_dofs = ch

    update_f!(f, values, prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, Kes, xmin, penalty, vars, varind, M)

    return
end

function update_f!(f::Vector{T}, values, prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, Kes, xmin, penalty, vars, varind, M) where {T}
    for ind in 1:length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(Kes[ind], 1)

        r = dof_cells.offsets[d] : dof_cells.offsets[d+1]-1
        if !applyzero && v != 0 
            for idx in r
                (i,j) = dof_cells.values[idx]
                px = ifelse(black[i], one(T), 
                            ifelse(white[i], xmin, 
                            px = penalty(density(vars[varind[i]], xmin))))
                if eltype(Kes) <: Symmetric
                    Ke = Kes[i].data
                else
                    Ke = Kes[i]
                end
                for row in 1:m
                    f[cell_dofs[row,i]] -= px * v * Ke[row,j]
                end
            end
        end
        f[d] = M*v
    end
    return 
end

function update_f!(f::CuVector{T}, values, prescribed_dofs, applyzero, dof_cells, cell_dofs, black, white, Kes, xmin, penalty, vars, varind, M) where {T}
    args = (f, values, prescribed_dofs, applyzero, dof_cells.offsets, dof_cells.values, cell_dofs, black, white, Kes, xmin, penalty, vars, varind, M)
    callkernel(dev, bc_kernel, args)
    CUDAdrv.synchronize(ctx)
    return 
end

function bc_kernel(f::AbstractVector{T}, values, prescribed_dofs, applyzero, dof_cells_offsets, dof_cells_values, cell_dofs, black, white, Kes, xmin, penalty, vars, varind, M) where {T}

    ind = @thread_global_index()
    offset = @total_threads()
    while ind <= length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(Kes[ind], 1)

        r = dof_cells_offsets[d] : dof_cells_offsets[d+1]-1
        if !applyzero && v != 0 
            for idx in r
                (i, j) = dof_cells_values[idx]
                px = ifelse(black[i], one(T), 
                        ifelse(white[i], xmin, 
                        penalty(density(vars[varind[i]], xmin))))
                if eltype(Kes) <: Symmetric
                    Ke = Kes[i].data
                else
                    Ke = Kes[i]
                end
                for row in 1:m
                    f[cell_dofs[row,i]] -= px * v * Ke[row,j]
                end
            end
        end
        f[d] = M*v
        ind += offset
    end
    return 
end
