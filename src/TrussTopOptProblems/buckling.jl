"""
    compute_local_axes(end_vert_u, end_vert_v)

# Arguments
`end_vert_u, end_vert_v` = vectors for nodal coordinate

# Outputs
`R` = (ndim x ndim) global_from_local transformation matrix
    Note that this matrix has its columns as axes
    So should be used as R*K*R' instead of R'*K*R as indicated in
    https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
"""
function compute_local_axes(end_vert_u, end_vert_v)
    @assert length(end_vert_u) == length(end_vert_v)
    @assert length(end_vert_u) == 2 || length(end_vert_u) == 3
    xdim = length(end_vert_u)
    L = norm(end_vert_u-end_vert_v)
    @assert L > eps()
    # by convention, the new x axis is along the element's direction
    # directional cosine of the new x axis in the global world frame
    c_x = (end_vert_v[1] - end_vert_u[1])/L
    c_y = (end_vert_v[2] - end_vert_u[2])/L
    R = zeros(xdim,xdim)
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
            new_y = -cross(new_x, [0,0,1.0])
            new_y /= norm(new_y)
            new_z = cross(new_x, new_y)
            R[:, 1] = new_x
            R[:, 2] = new_y
            R[:, 3] = new_z
        end
    elseif 2 == xdim
        R = [c_x  -c_y;
             c_y  c_x]
    end
    return R
end

"""
    get_truss_Kσs(problem::TrussProblem{xdim, TT}, u, cellvalues) where {xdim, TT}

Compute the geometric stiffness matrix for **truss elements** (axial bar element, no bending or torsion). Matrix formulation defined in eq (3) and (4) in
https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf

This matrix formulation is equivalent to the one used in 
    M. Kočvara, “On the modelling and solving of the truss design problem with global stability constraints,” Struct Multidisc Optim, vol. 23, no. 3, pp. 189–203, Apr. 2002, doi: 10/br35mf.

# Arguments
`problem` = truss topopt problem struct
`u` = deformation vector (solved from first-order linear elasticity)
`cellvalues` = Ferrite cellvalues of the truss system

# Outputs
`Kσs` = a list of 2*ndim x 2*ndim element geometric stiffness matrix (in global cooridnate)
"""
function get_truss_Kσs(problem::TrussProblem{xdim, TT}, u, cellvalues) where {xdim, TT}
    Es = getE(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert ndof_pc == xdim*n_basefuncs "$ndof_pc, $n_basefuncs"
    @assert n_basefuncs == 2

    global_dofs = zeros(Int, ndof_pc)
    Kσs = [zeros(TT, ndof_pc, ndof_pc) for i in 1:getncells(dh.grid)]
    Kσ_e = zeros(TT, ndof_pc, ndof_pc)

    for (cellidx, cell) in enumerate(CellIterator(dh))
        Kσ_e .= 0
        truss_reinit!(cellvalues, cell, As[cellidx])
        # get cell's dof's global dof indices, i.e. CC_a^e
        celldofs!(global_dofs, dh, cellidx)
        E = Es[cellidx]
        A = As[cellidx]
        L = norm(cell.coords[1] - cell.coords[2])
        R = compute_local_axes(cell.coords[1], cell.coords[2])
        # local axial projection operator (local axis transformation)
        γ = vcat(-R[:,1], R[:,1])
        u_cell = @view u[global_dofs]
        # compute axial force: first-order approx of bar force
        # better approx would be: EA/L * (u3-u1 + 1/(2*L0)*(u4-u2)^2) = EA/L * (γ'*u + 1/2*(δ'*u)^2)
        # see: https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
        q_cell = E*A/L*(γ'*u_cell)
        for i=2:size(R,2)
            δ = vcat(-R[:,i], R[:,i])
            # @assert δ' * γ ≈ 0
            Kσ_e .+= δ * δ'
        end
        Kσ_e .*= q_cell / L
        Kσs[cellidx] .= Kσ_e
    end
    return Kσs
end

"""
    buckling(problem::TrussProblem{xdim, T}, ginfo, einfo, vars=ones(T, getncells(getdh(problem).grid)), xmin = T(0.0); u=undef) where {xdim, T}

Assembly global geometric stiffness matrix of the given truss problem.

# Arguments
`problem` = truss topopt problem
`ginfo` = solver.globalinfo 
`einfo` = solver.elementinfo 
`vars` = (Optional) design variable values, default to all ones
`xmin` = (Optional) min x value, default to 0.0
`u` = (Optional) given displacement vector, if specified as undef, a linear solve will be performed on the first-order K to get u

# Outputs
`K` = global first-order stiffness matrix (same as ginfo.K)
`Kσ` = global geometric stiffness matrix
"""
function buckling(problem::TrussProblem{xdim, T}, ginfo, einfo, vars=ones(T, getncells(getdh(problem).grid)), xmin = T(0.0); u=undef) where {xdim, T}
    dh = problem.ch.dh
    black = problem.black
    white = problem.white
    varind = problem.varind # variable index from cell index

    if u === undef
        u = ginfo.K \ ginfo.f
    end
    Kσs = get_truss_Kσs(problem, u, einfo.cellvalues)
    Kσ = deepcopy(ginfo.K)

    if Kσ isa Symmetric
        Kσ.data.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(Kσ.data, T[], Int[], Int[])
    else
        Kσ.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(Kσ, T[], Int[], Int[])
    end

    # * assemble global geometric stiffness matrix
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    Kσ_e = zeros(T, size(Kσs[1]))
    celliteratortype = CellIterator{typeof(dh).parameters...}
    celliterator::celliteratortype = CellIterator(dh)
    TK = eltype(Kσs)
    for (i,cell) in enumerate(celliterator)
        celldofs!(global_dofs, dh, i)
        Kσ_e = TK isa Symmetric ? Kσs[i].data : Kσs[i]
        if black[i]
            Ferrite.assemble!(assembler, global_dofs, Kσ_e)
        elseif white[i]
            # if PENALTY_BEFORE_INTERPOLATION
            px = xmin
            # else
            #     px = penalty(xmin)
            # end
            Kσ_e = px * Kσ_e
            Ferrite.assemble!(assembler, global_dofs, Kσ_e)
        else
            px = vars[varind[i]]
            # if PENALTY_BEFORE_INTERPOLATION
            # px = density(penalty(vars[varind[i]]), xmin)
            # else
                # px = penalty(density(vars[varind[i]], xmin))
            # end
            Kσ_e = px * Kσ_e
            Ferrite.assemble!(assembler, global_dofs, Kσ_e)
        end
    end

    #* apply boundary condition
    Kσ = _apply!(Kσ, problem.ch)

    return ginfo.K, Kσ
end

using ChainRulesCore

function _apply!(Kσ, ch)
    apply!(Kσ, eltype(Kσ)[], ch, true)
    return Kσ
end
function ChainRulesCore.rrule(::typeof(_apply!), Kσ, ch)
    project_to = ChainRulesCore.ProjectTo(Kσ)
    return _apply!(Kσ, ch), Δ -> begin
        NoTangent(), _apply!(project_to(Δ), ch) , NoTangent()
    end
end
