function matrix_free_apply2f!(
    f::AbstractVector{T},
    elementinfo::ElementFEAInfo{dim,T},
    M,
    vars,
    problem::StiffnessTopOptProblem,
    penalty,
    xmin,
    applyzero::Bool = false,
) where {dim,T}
    @unpack Kes, black, white, varind, metadata = elementinfo
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
        black,
        white,
        Kes,
        xmin,
        penalty,
        vars,
        varind,
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
    black,
    white,
    Kes,
    xmin,
    penalty,
    vars,
    varind,
    M,
) where {T}
    for ind = 1:length(values)
        d = prescribed_dofs[ind]
        v = values[ind]
        m = size(Kes[ind], 1)

        r = dof_cells.offsets[d]:(dof_cells.offsets[d+1]-1)
        if !applyzero && v != 0
            for idx in r
                (i, j) = dof_cells.values[idx]
                if PENALTY_BEFORE_INTERPOLATION
                    px = ifelse(
                        black[i],
                        one(T),
                        ifelse(
                            white[i],
                            xmin;
                            px = density(penalty(vars[varind[i]]), xmin),
                        ),
                    )
                else
                    px = ifelse(
                        black[i],
                        one(T),
                        ifelse(
                            white[i],
                            xmin;
                            px = penalty(density(vars[varind[i]], xmin)),
                        ),
                    )
                end
                if eltype(Kes) <: Symmetric
                    Ke = Kes[i].data
                else
                    Ke = Kes[i]
                end
                for row = 1:m
                    f[cell_dofs[row, i]] -= px * v * Ke[row, j]
                end
            end
        end
        f[d] = M * v
    end
    return nothing
end
