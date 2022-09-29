using ..TopOpt: ElementFEAInfo, ElementMatrix
using ..TopOpt.TopOptProblems: make_cload, convert

"""
    ElementFEAInfo(sp, quad_order=2, ::Type{Val{mat_type}}=Val{:Static}) where {mat_type}

Constructs an instance of `ElementFEAInfo` from a stiffness **truss** problem `sp` using a Gaussian quadrature order of `quad_order`. The element matrix and vector types will be:
1. `SMatrix` and `SVector` if `mat_type` is `:SMatrix` or `:Static`, the default,
2. `MMatrix` and `MVector` if `mat_type` is `:MMatrix`, or
3. `Matrix` and `Vector` otherwise.

The static matrices and vectors are more performant and GPU-compatible therefore they are used by default.
"""
function ElementFEAInfo(
    sp::TrussProblem,
    quad_order = 1,
    ::Type{Val{mat_type}} = Val{:Static},
) where {mat_type}
    # weights: self-weight element load vectors, all zeros now
    Kes, weights, cellvalues, facevalues = make_Kes_and_fes(sp, quad_order, Val{mat_type})
    element_Kes = convert(
        Vector{<:ElementMatrix},
        Kes;
        bc_dofs = sp.ch.prescribed_dofs,
        dof_cells = sp.metadata.dof_cells,
    )

    # * concentrated load
    # ? why convert a sparse vector back to a Vector?
    fixedload = Vector(make_cload(sp))
    # * distributed load (if any)
    # assemble_f!(fixedload, sp, dloads)

    cellvolumes = get_cell_volumes(sp, cellvalues)
    cells = sp.ch.dh.grid.cells
    return ElementFEAInfo(
        element_Kes,
        weights,
        fixedload,
        cellvolumes,
        cellvalues,
        facevalues,
        sp.metadata,
        sp.black,
        sp.white,
        sp.varind,
        cells,
    )
end

####################################

function get_cell_volumes(sp::TrussProblem{xdim,T}, cellvalues) where {xdim,T}
    dh = sp.ch.dh
    As = getA(sp)
    cellvolumes = zeros(T, getncells(dh.grid))
    for (i, cell) in enumerate(CellIterator(dh))
        truss_reinit!(cellvalues, cell, As[i])
        cellvolumes[i] = sum(
            Ferrite.getdetJdV(cellvalues, q_point) for
            q_point = 1:Ferrite.getnquadpoints(cellvalues)
        )
    end
    return cellvolumes
end

####################################

"""
    compute_local_axes(end_vert_u, end_vert_v)

# Arguments
`end_vert_u, end_vert_v` = vectors for nodal coordinate

# Outputs
`R` = (ndim x ndim) global_from_local transformation matrix
    Each column of the matrix is the local system's axis coordinate in the world system.
    Note that this matrix has its columns as axes
    So should be used as `R*K*R'` instead of `R'*K*R` as indicated in
    https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
"""
function compute_local_axes(end_vert_u, end_vert_v)
    @assert length(end_vert_u) == length(end_vert_v)
    @assert length(end_vert_u) == 2 || length(end_vert_u) == 3
    xdim = length(end_vert_u)
    L = norm(end_vert_u - end_vert_v)
    @assert L > eps()
    # by convention, the new x axis is along the element's direction
    # directional cosine of the new x axis in the global world frame
    c_x = (end_vert_v[1] - end_vert_u[1]) / L
    c_y = (end_vert_v[2] - end_vert_u[2]) / L
    R = zeros(xdim, xdim)
    if 3 == xdim
        c_z = (end_vert_v[3] - end_vert_u[3]) / L
        if abs(abs(c_z) - 1.0) < eps()
            R[1, 3] = -c_z
            R[2, 2] = 1
            R[3, 1] = c_z
        else
            # local x_axis = element's vector
            new_x = [c_x, c_y, c_z]
            # local y axis = cross product with global z axis
            new_y = -cross(new_x, [0, 0, 1.0])
            new_y /= norm(new_y)
            new_z = cross(new_x, new_y)
            R[:, 1] = new_x
            R[:, 2] = new_y
            R[:, 3] = new_z
        end
    elseif 2 == xdim
        R = [
            c_x -c_y
            c_y c_x
        ]
    end
    return R
end
