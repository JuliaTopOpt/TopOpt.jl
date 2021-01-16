using Einsum

function get_Kσs(sp::StiffnessTopOptProblem{dim, TT}, dofs, cellvalues) where {dim, TT}
    E = getE(sp)
    ν = getν(sp)
    dh = sp.ch.dh
    n = ndofs_per_cell(dh)
    global_dofs = zeros(Int, n)
    Kσs = [zeros(TT, n, n) for i in 1:getncells(dh.grid)]
    Kσ_e = zeros(TT, n, n)
    ψ_e = zeros(TT, dim*n, dim*n)
    G = zeros(TT, dim*n, n)
    δ = Matrix(TT(1.0)I, dim, dim)
    ϵ = zeros(TT, dim, dim)
    σ = zeros(TT, dim, dim)
    u = zeros(TT, dim, dim)
    n_basefuncs = getnbasefunctions(cellvalues)
    for (cellidx, cell) in enumerate(CellIterator(dh))
        Kσ_e .= 0
        reinit!(cellvalues, cell)
        celldofs!(global_dofs, dh, cellidx)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for d in 1:dim
                ψ_e[(d-1)*dim+1:d*dim, (d-1)*dim+1:d*dim] .= 0
            end
            for a in 1:n_basefuncs
                ∇ϕ = shape_gradient(cellvalues, q_point, a)
                _u = @view dofs[(@view global_dofs[dim*(a-1) .+ (1:dim)])]
                @einsum u[i,j] = _u[i]*∇ϕ[j]
                @einsum ϵ[i,j] = 1/2*(u[i,j] + u[j,i])
                @einsum σ[i,j] = E*ν/(1-ν^2)*δ[i,j]*ϵ[k,k] + E*ν*(1+ν)*ϵ[i,j]
                for d in 1:dim
                    ψ_e[(d-1)*dim+1:d*dim, (d-1)*dim+1:d*dim] .+= σ
                    G[(dim*(d-1)+1):(dim*d), (a-1)*dim+d] .= ∇ϕ
                end
            end
            Kσ_e .+= G'*ψ_e*G*dΩ
        end
        Kσs[cellidx] .= Kσ_e
    end

    return Kσs
end

function buckling(problem::StiffnessTopOptProblem{dim, T}, ginfo, einfo) where {dim, T}
    dh = problem.ch.dh

    u = ginfo.K \ ginfo.f
    Kσs = get_Kσs(problem, u, einfo.cellvalues)
    Kσ = deepcopy(ginfo.K)

    if Kσ isa Symmetric
        Kσ.data.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(Kσ.data, T[], Int[], Int[])
    else
        Kσ.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(Kσ, T[], Int[], Int[])
    end

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

    return ginfo.K, Kσ
end

