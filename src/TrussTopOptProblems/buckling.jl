# modified from https://github.com/mohamed82008/LinearElasticity.jl
using Einsum: @einsum
using LinearAlgebra: I, norm

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
        R[:,1] = [c_x, c_y]
        R[:,2] = [-c_y, c_x]
    end
    return R
end

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
        γ = vcat(R[:,1], -R[:,1])
        # compute axial force
        u_cell = @view u[global_dofs]
        q_cell = E*A*(γ'*u_cell/L)
        for i=2:size(R,2)
            δ = vcat(R[:,i], -R[:,i])
            @assert δ' * γ ≈ 0
            Kσ_e .+= q_cell / L^2 .* δ * δ'
        end
        # ? why do we need the negative sign here?
        Kσs[cellidx] .= -Kσ_e
    end
    return Kσs
end

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
    apply!(Kσ, ginfo.f, problem.ch)

    return ginfo.K, Kσ
end

##########################################
# TODO haven't figured out why this doesn't work
# sometimes Kσ_e = 0 
function get_Kσs(problem::TrussProblem{xdim, TT}, u, cellvalues) where {xdim, TT}
    Es = getE(problem)
    νs = getν(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert ndof_pc == xdim*n_basefuncs "$ndof_pc, $n_basefuncs"

    global_dofs = zeros(Int, ndof_pc)
    Kσs = [zeros(TT, ndof_pc, ndof_pc) for i in 1:getncells(dh.grid)]
    Kσ_e = zeros(TT, ndof_pc, ndof_pc)
    # block-diagonal - block σ_e = σ_ij, i,j in xdim
    ψ_e = zeros(TT, xdim*xdim, xdim*xdim)
    G = zeros(TT, xdim*xdim, xdim*n_basefuncs)
    δ = Matrix(TT(1.0)I, xdim, xdim)
    ϵ = zeros(TT, xdim, xdim)
    σ = zeros(TT, xdim, xdim)
    # u_i,j: partial derivative
    u_p = zeros(TT, xdim, xdim)

    for (cellidx, cell) in enumerate(CellIterator(dh))
        Kσ_e .= 0
        truss_reinit!(cellvalues, cell, As[cellidx])
        # get cell's dof's global dof indices, i.e. CC_a^e
        celldofs!(global_dofs, dh, cellidx)
        E = Es[cellidx]
        ν = νs[cellidx]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for d in 1:xdim
                ψ_e[(d-1)*xdim+1:d*xdim, (d-1)*xdim+1:d*xdim] .= 0
            end
            for a in 1:n_basefuncs
                ∇ϕ = shape_gradient(cellvalues, q_point, a)
                # given displacement values of the cell nodes
                u_cell = @view u[(@view global_dofs[xdim*(a-1) .+ (1:xdim)])]
                # u_i,j, i for spatial xdim, j for partial derivative
                @einsum u_p[i,j] = u_cell[i]*∇ϕ[j]
                # linear strain: effect of the quadratic term in the strain formula have on the stress field is ignored
                @einsum ϵ[i,j] = 1/2*(u_p[i,j] + u_p[j,i])
                # ! truss element special treatment here
                # isotropic solid
                # @einsum σ[i,j] = E*ν/(1-ν^2)*δ[i,j]*ϵ[k,k] + E*ν*(1+ν)*ϵ[i,j]
                # σ = E .* ϵ
                σ = E .* ones(size(ϵ))/2
                
                for d in 1:xdim
                    ψ_e[(d-1)*xdim+1:d*xdim, (d-1)*xdim+1:d*xdim] .+= σ
                    G[(xdim*(d-1)+1):(xdim*d), (a-1)*xdim+d] .= ∇ϕ
                end
            end
            @show G
            @show ψ_e
            Kσ_e .+= G'*ψ_e*G*dΩ
        end
        @show Kσ_e
        Kσs[cellidx] .= Kσ_e
    end
    return Kσs
end

