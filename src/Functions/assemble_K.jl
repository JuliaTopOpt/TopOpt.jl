@params mutable struct AssembleK{T} <: AbstractFunction{T}
	problem::StiffnessTopOptProblem{T}
    K::AbstractMatrix{T}
    x::AbstractVector{T} # topology design variable vector
    global_dofs::AbstractVector{<:Integer} # preallocated dof vector for a cell
end

Base.show(::IO, ::MIME{Symbol("text/plain")}, ::AssembleK) = println("TopOpt global linear stiffness matrix assembly function")

function AssembleK(problem::StiffnessTopOptProblem{T}, x::AbstractVector{T}) where {T}
    dh = problem.ch.dh
    k = ndofs_per_cell(dh)
    global_dofs = zeros(Int, k)
    return AssembleK(problem, sparse(zeros(T, 0, 0)), x, global_dofs)
end

function (ak::AssembleK{T})(Kes::AbstractVector{<:AbstractMatrix{T}}) where {T}
    @unpack problem, K, x, global_dofs = ak
    dh = solver.problem.ch.dh

    if K isa Symmetric
        K.data.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(K.data, T[], Int[], Int[])
    else
        K.nzval .= 0
        assembler = Ferrite.AssemblerSparsityPattern(K, T[], Int[], Int[])
    end

    Ke = zeros(T, size(Kes[1]))
    TK = eltype(Kes)
    for (i,cell) in enumerate(CellIterator(dh))
        celldofs!(global_dofs, dh, i)
        Ke = TK isa Symmetric ? Kes[i].data : Kes[i]
        if black[i]
            Ferrite.assemble!(assembler, global_dofs, Ke)
        elseif white[i]
            px = xmin
            Ke = px * Ke
            Ferrite.assemble!(assembler, global_dofs, Ke)
        else
            px = x[varind[i]]
            Ke = px * Ke
            Ferrite.assemble!(assembler, global_dofs, Ke)
        end
    end

    return copy(K)
end

"""
rrule for autodiff (takes output space wobbles, gives input space wiggles)

assembleK(Kes) = (Σ_ei x_e * K_ei)

d(K)/d(Ke) = 

where d(K)/d(Ke) ∈ ((global K) x (list of Ke)); 
d(K)/d(Ke)^T * Δ = ((list of Ke) x (global K)) * Δ(global K) -> list of Ke
"""
function ChainRulesCore.rrule(ak::AssembleK, Kes)
    @unpack problem, x, global_dofs = ak
    @unpack metadata = problem
    # dof_cells[dofidx] = [(cellidx, cell's local dof idx), ...]
    @unpack dof_cells = metadata
    dh = solver.problem.ch.dh
    K = ak(Kes)

    n_dofs = length(global_dofs)
    tmp_Kes = similar(Kes)
    for tmp_Ke in tmp_Kes
        fill!(tmp_Ke, 0.0)
    end

    function assembleK_pullback(Δ)
        # I, J, V = findnz(Δ)
        # for (i,j,v) in zip(I,J,V)
        #     i_dof_cells = dof_cells[i] # all the cells that share dof i
        #     j_dof_cells = dof_cells[j] # all the cells that share dof j
        #     # take intersection of i_cells and j_cells
        # end
        for (ci,cell) in enumerate(CellIterator(dh))
            tmp_Kes[ci] .= 0.0
            celldofs!(global_dofs, dh, i)
            for i in 1:n_dofs
                for j in 1:n_dofs
                    # ! should we do an average over all the element matrices that share the dof?
                    tmp_Kes[ci][i,j] = Δ[global_dofs[i], global_dofs[j]]
                end
            end
        end
        return nothing, tmp_Kes
    end

    return K, assembleK_pullback
end
