# modified from https://github.com/mohamed82008/LinearElasticity.jl
using Einsum: @einsum
using LinearAlgebra: I

function get_Kσs(problem::TrussProblem{xdim, TT}, u_dofs, cellvalues) where {xdim, TT}
    Es = getE(problem)
    νs = getν(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    # ? number of nodes per cell == n_basefuncs per cell
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
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
                _u = @view u_dofs[(@view global_dofs[xdim*(a-1) .+ (1:xdim)])]
                # u_i,j, i for spatial xdim, j for partial derivative
                @einsum u_p[i,j] = _u[i]*∇ϕ[j]
                # effect of the quadratic term in the strain formula have on the stress field is ignored
                @einsum ϵ[i,j] = 1/2*(u_p[i,j] + u_p[j,i])
                # isotropic solid
                # @einsum σ[i,j] = E*ν/(1-ν^2)*δ[i,j]*ϵ[k,k] + E*ν*(1+ν)*ϵ[i,j]
                # ! truss element special treatment here
                σ = E .* ϵ
                
                for d in 1:xdim
                    ψ_e[(d-1)*xdim+1:d*xdim, (d-1)*xdim+1:d*xdim] .+= σ
                    G[(xdim*(d-1)+1):(xdim*d), (a-1)*xdim+d] .= ∇ϕ
                end
            end
            # @show G
            Kσ_e .+= G'*ψ_e*G*dΩ
        end
        # @show Kσ_e
        Kσs[cellidx] .= Kσ_e
    end
    return Kσs
end

function buckling(problem::TrussProblem{xdim, T}, ginfo, einfo; u=undef) where {xdim, T}
    dh = problem.ch.dh

    if u === undef
        u = ginfo.K \ ginfo.f
    end
    Kσs = get_Kσs(problem, u, einfo.cellvalues)
    Kσ = deepcopy(ginfo.K)

    # @show Kσs

    if Kσ isa Symmetric
        Kσ.data.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(Kσ.data, T[], Int[], Int[])
    else
        Kσ.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(Kσ, T[], Int[], Int[])
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
            JuAFEM.assemble!(assembler, global_dofs, Kσs[i].data)
        else
            JuAFEM.assemble!(assembler, global_dofs, Kσs[i])
        end
    end

    # f = copy(ginfo.f)
    # apply!(Kσ, f, problem.ch)

    return ginfo.K, Kσ
end
