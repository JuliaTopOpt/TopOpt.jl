using Einsum

function get_Kσs(sp::StiffnessTopOptProblem{xdim, TT}, u_dofs, cellvalues) where {xdim, TT}
    E = getE(sp)
    ν = getν(sp)
    dh = sp.ch.dh
    # usually ndof_pc = xdim * n_basefuncs
    # ? number of nodes per cell == n_basefuncs per cell
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndof_pc)
    Kσs = [zeros(TT, ndof_pc, ndof_pc) for i in 1:getncells(dh.grid)]
    Kσ_e = zeros(TT, ndof_pc, ndof_pc)
    # block-diagonal - block σ_e = σ_ij, i,j in xdim
    # ! shouldn't this be xdim*xdim by xdim*xdim?
    # ? ψ_e = zeros(TT, xdim*ndof_pc, xdim*ndof_pc)
    ψ_e = zeros(TT, xdim*xdim, xdim*xdim)
    # ? G = zeros(TT, xdim*ndof_pc, ndof_pc)
    G = zeros(TT, xdim*xdim, xdim*n_basefuncs)
    δ = Matrix(TT(1.0)I, xdim, xdim)
    ϵ = zeros(TT, xdim, xdim)
    σ = zeros(TT, xdim, xdim)
    # u_i,j: partial derivative
    u_p = zeros(TT, xdim, xdim)
    for (cellidx, cell) in enumerate(CellIterator(dh))
        Kσ_e .= 0
        reinit!(cellvalues, cell)
        # get cell's dof's global dof indices, i.e. CC_a^e
        celldofs!(global_dofs, dh, cellidx)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for d in 1:xdim
                ψ_e[(d-1)*xdim+1:d*xdim, (d-1)*xdim+1:d*xdim] .= 0
            end
            for a in 1:n_basefuncs
                ∇ϕ = shape_gradient(cellvalues, q_point, a)
                _u = @view u_dofs[(@view global_dofs[xdim*(a-1) .+ (1:xdim)])]
                # u_i,j, i for spatial xdim, j for partial derivative
                @einsum u_p[i,j] = _u[i]*∇ϕ[j]
                # effect of the quadratic term in the strain formula have on the stress field is ignored
                @einsum ϵ[i,j] = 1/2*(u_p[i,j] + u_p[j,i])
                # isotropic solid
                @einsum σ[i,j] = E*ν/(1-ν^2)*δ[i,j]*ϵ[k,k] + E*ν*(1+ν)*ϵ[i,j]
                for d in 1:xdim
                    # block diagonal
                    ψ_e[(d-1)*xdim .+ 1:d*xdim, (d-1)*xdim .+ 1:d*xdim] .+= σ
                    G[(xdim*(d-1)+1):(xdim*d), (a-1)*xdim+d] .= ∇ϕ
                end
            end
            Kσ_e .+= G'*ψ_e*G*dΩ
        end
        Kσs[cellidx] .= Kσ_e
    end

    return Kσs
end

function buckling(problem::StiffnessTopOptProblem{xdim, T}, ginfo, einfo) where {xdim, T}
    dh = problem.ch.dh

    u = ginfo.K \ ginfo.f
    Kσs = get_Kσs(problem, u, einfo.cellvalues)
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
    _celliterator::celliteratortype = CellIterator(dh)
    TK = eltype(Kσs)
    for (i,cell) in enumerate(_celliterator)
        celldofs!(global_dofs, dh, i)
        if TK <: Symmetric
            Ferrite.assemble!(assembler, global_dofs, Kσs[i].data)
        else
            Ferrite.assemble!(assembler, global_dofs, Kσs[i])
        end
    end

    return ginfo.K, Kσ
end

