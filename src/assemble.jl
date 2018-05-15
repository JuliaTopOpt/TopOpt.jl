function assemble(problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}, vars, penalty, xmin = T(1)/1000) where {dim,T}
    globalinfo = GlobalFEAInfo(problem)
    assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
    return globalinfo
end

function assemble!(globalinfo::GlobalFEAInfo{T}, problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T, TK}, vars, penalty, xmin = T(1)/1000) where {dim, T, TK}
    ch = problem.ch
    dh = ch.dh
    K, f = globalinfo.K, globalinfo.f
    f .= elementinfo.fixedload

    Kes, fes = elementinfo.Kes, elementinfo.fes
    black = problem.black
    white = problem.white
    varind = problem.varind

    if K isa Symmetric
        K.data.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(K.data, f, Int[], Int[])
    else
        K.nzval .= 0
        assembler = JuAFEM.AssemblerSparsityPattern(K, f, Int[], Int[])
    end

    global_dofs = zeros(Int, ndofs_per_cell(dh))
    fe = zeros(fes[1])
    Ke = zeros(T, size(Kes[1]))

    celliteratortype = CellIterator{typeof(dh).parameters...}
    _celliterator::celliteratortype = CellIterator(dh)
    for (i,cell) in enumerate(_celliterator)
        celldofs!(global_dofs, dh, i)
        if black[i]
            if TK <: Symmetric
                JuAFEM.assemble!(assembler, global_dofs, Kes[i].data, fes[i])
            else
                JuAFEM.assemble!(assembler, global_dofs, Kes[i], fes[i])
            end
        elseif white[i]
            px = penalty(xmin)
            fe .= px .* fes[i]
            if TK <: Symmetric
                Ke .= px .* Kes[i].data
            else
                Ke .= px .* Kes[i]
            end
            JuAFEM.assemble!(assembler, global_dofs, Ke, fe)
        else
            px = penalty(density(vars[varind[i]], xmin))
            fe .= px .* fes[i]
            if TK <: Symmetric
                Ke .= px .* Kes[i].data
            else
                Ke .= px .* Kes[i]
            end
            JuAFEM.assemble!(assembler, global_dofs, Ke, fe)
        end
    end

    if TK <: Symmetric
        apply!(K.data, f, ch)
    else
        apply!(K, f, ch)
    end

    return 
end

function assemble_f(problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}, vars::AbstractVector{T}, penalty, xmin = T(1)/1000) where {dim, T}
    f = zeros(T, ndofs(problem.ch.dh))
    assemble_f!(f, problem, elementinfo, vars, penalty, xmin)
    return f
end
function assemble_f!(f::AbstractVector, problem::StiffnessTopOptProblem, elementinfo::ElementFEAInfo, vars::AbstractVector, penalty, xmin)
    black = problem.black
    white = problem.white
    varind = problem.varind
    fes = elementinfo.fes

    f .= elementinfo.fixedload
    dof_cells = problem.metadata.dof_cells
    dof_cells_offset = problem.metadata.dof_cells_offset
    for dofidx in 1:ndofs(problem.ch.dh)
        r = dof_cells_offset[dofidx] : dof_cells_offset[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells[i]
            if black[cellidx]
                f[dofidx] += fes[cellidx][localidx]
            elseif white[cellidx]
                px = penalty(xmin)
                f[dofidx] += px * fes[cellidx][localidx]                
            else
                px = penalty(density(vars[varind[cellidx]], xmin))
                f[dofidx] += px * fes[cellidx][localidx]                
            end
        end
    end
    return f
end
