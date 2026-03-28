function matrix_free_apply2f!(
    f::AbstractVector{T},
    elementinfo::ElementFEAInfo{dim,T},
    M,
    vars,
    problem::StiffnessTopOptProblem,
    penalty,
    xmin,
    applyzero::Bool=false,
) where {dim,T}
    @unpack Kes, metadata = elementinfo
    @unpack dof_cells, cell_dofs = metadata
    @unpack ch = problem
    @unpack values, prescribed_dofs = ch

    update_f!(
        f,
        values,
        prescribed_dofs,
        applyzero,
        dof_cells,
        cell_dofs,
        Kes,
        xmin,
        penalty,
        vars,
        M,
    )

    return nothing
end

function update_f!(
    f::Vector{T},
    values,
    prescribed_dofs,
    applyzero,
    dof_cells,
    cell_dofs,
    Kes,
    xmin,
    penalty,
    vars,
        M,
) where {T}
    for ind in 1:length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(Kes[ind], 1)

        r = dof_cells.offsets[d]:(dof_cells.offsets[d + 1] - 1)
        if !applyzero && v != 0
            for idx in r
                (i, j) = dof_cells.values[idx]
                if PENALTY_BEFORE_INTERPOLATION
                    px = density(penalty(vars[i]), xmin)
                else
                    px = penalty(density(vars[i], xmin))
                end
                Ke = Kes[i].data
                for row in 1:m
                    f[cell_dofs[row, i]] -= px * v * Ke[row, j]
                end
            end
        end
        f[d] = M * v
    end
    return nothing
end
