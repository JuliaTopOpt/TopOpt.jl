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
function ElementFEAInfo(sp::TrussProblem, quad_order = 2, ::Type{Val{mat_type}} = Val{:Static},) where {mat_type} 
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
    ElementFEAInfo(
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

function get_cell_volumes(sp::TrussProblem{xdim, T}, cellvalues) where {xdim, T}
    dh = sp.ch.dh
    As = getA(sp)
    cellvolumes = zeros(T, getncells(dh.grid))
    for (i, cell) in enumerate(CellIterator(dh))
        truss_reinit!(cellvalues, cell, As[i])
        cellvolumes[i] = sum(JuAFEM.getdetJdV(cellvalues, q_point) for q_point in 1:JuAFEM.getnquadpoints(cellvalues))
    end
    return cellvolumes
end

