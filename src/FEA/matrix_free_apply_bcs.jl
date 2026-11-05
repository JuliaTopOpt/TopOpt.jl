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
    # Ferrite 1.x renamed the prescribed-BC-value field from `values` to
    # `inhomogeneities` (one entry per prescribed dof).
    @unpack prescribed_dofs = ch
    values = ch.inhomogeneities

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
    # `values` is indexed per prescribed DOF while `Kes` is indexed per cell;
    # all cells share the same number of local DOFs, so the local size comes
    # from the first cell's matrix.
    m = size(first(Kes), 1)
    for ind in eachindex(values)
        d = prescribed_dofs[ind]
        v = values[ind]

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
