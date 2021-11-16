using Ferrite
using Ferrite: getnbasefunctions, CellIterator
using TopOpt.TrussTopOptProblems.TrussTopOptProblems: compute_local_axes, getE, getA
using TopOpt.TrussTopOptProblems: truss_reinit!

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
        Kσ_e .= 0
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
    # TODO replace
    # assemble_k = TopOpt.AssembleK(problem)
    # Kσ = assemble_k(Kσs)

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
    Kσ = apply_boundary_with_zerodiag!(Kσ, problem.ch)

    return ginfo.K, Kσ
end