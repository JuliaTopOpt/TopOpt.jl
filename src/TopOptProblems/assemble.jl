function assemble(problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}, vars = ones(T, getncells(getdh(problem).grid)), penalty = PowerPenalty(T(1)), xmin = T(0.001)) where {dim,T}
    globalinfo = GlobalFEAInfo(problem)
    assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
    return globalinfo
end

function assemble!(globalinfo::GlobalFEAInfo{T}, problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T, TK}, vars = ones(T, getncells(getdh(problem).grid)), penalty = PowerPenalty(T(1)), xmin = T(0.001); assemble_f = true) where {dim, T, TK}
    ch = problem.ch
    dh = ch.dh
    K, f = globalinfo.K, globalinfo.f
    if assemble_f
        f .= elementinfo.fixedload
    end
    Kes, fes = elementinfo.Kes, elementinfo.fes
    black = problem.black
    white = problem.white
    varind = problem.varind

    _K = K isa Symmetric ? K.data : K
    _K.nzval .= 0
    assembler = Ferrite.AssemblerSparsityPattern(_K, f, Int[], Int[])

    global_dofs = zeros(Int, ndofs_per_cell(dh))
    fe = zeros(typeof(fes[1]))
    Ke = zeros(T, size(rawmatrix(Kes[1])))

    celliterator = CellIterator(dh)
    for (i,cell) in enumerate(celliterator)
        celldofs!(global_dofs, dh, i)
        fe = fes[i]
        _Ke = rawmatrix(Kes[i])
        Ke = _Ke isa Symmetric ? _Ke.data : _Ke
        if black[i]
            if assemble_f
                Ferrite.assemble!(assembler, global_dofs, Ke, fe)
            else
                Ferrite.assemble!(assembler, global_dofs, Ke)
            end
        elseif white[i]
            if PENALTY_BEFORE_INTERPOLATION
                px = xmin
            else
                px = penalty(xmin)
            end
            Ke = px * Ke
            if assemble_f
                fe = px * fe
                Ferrite.assemble!(assembler, global_dofs, Ke, fe)
            else
                Ferrite.assemble!(assembler, global_dofs, Ke)
            end
        else
            if PENALTY_BEFORE_INTERPOLATION
                px = density(penalty(vars[varind[i]]), xmin)
            else
                px = penalty(density(vars[varind[i]], xmin))
            end
            Ke = px * Ke
            if assemble_f
                fe = px * fe
                Ferrite.assemble!(assembler, global_dofs, Ke, fe)
            else
                Ferrite.assemble!(assembler, global_dofs, Ke)
            end
        end
    end

    _K = TK <: Symmetric ? K.data : K        
    apply!(_K, f, ch)

    return 
end

function assemble_f(problem::StiffnessTopOptProblem{dim,T}, elementinfo::ElementFEAInfo{dim, T}, vars::AbstractVector{T}, penalty, xmin = T(1)/1000) where {dim, T}
    f = get_f(problem, vars)
    assemble_f!(f, problem, elementinfo, vars, penalty, xmin)
    return f
end
get_f(problem, vars::Array) = zeros(T, ndofs(problem.ch.dh))

function assemble_f!(f::AbstractVector, problem::StiffnessTopOptProblem, 
        elementinfo::ElementFEAInfo, vars::AbstractVector, penalty, xmin)
    black = elementinfo.black
    white = elementinfo.white
    varind = elementinfo.varind
    fes = elementinfo.fes

    dof_cells = elementinfo.metadata.dof_cells

    update_f!(f, fes, elementinfo.fixedload, dof_cells, black, 
        white, penalty, vars, varind, xmin)
    return f
end

function update_f!(f::Vector, fes, fixedload, dof_cells, black, 
    white, penalty, vars, varind, xmin)

    @inbounds for dofidx in 1:length(f)
        f[dofidx] = fixedload[dofidx]
        r = dof_cells.offsets[dofidx] : dof_cells.offsets[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells.values[i]
            if black[cellidx]
                f[dofidx] += fes[cellidx][localidx]
            elseif white[cellidx]
                if PENALTY_BEFORE_INTERPOLATION
                    px = xmin
                else
                    px = penalty(xmin)
                end
                f[dofidx] += px * fes[cellidx][localidx]                
            else
                if PENALTY_BEFORE_INTERPOLATION
                    px = density(penalty(vars[varind[cellidx]]), xmin)
                else
                    px = penalty(density(vars[varind[cellidx]], xmin))
                end
                f[dofidx] += px * fes[cellidx][localidx]                
            end
        end
    end

    return
end

function assemble_f!(f::AbstractVector, problem, dloads)
    metadata = problem.metadata
    dof_cells = metadata.dof_cells
    update_f!(f, dof_cells, dloads)
    return f
end

function update_f!(f::Vector, dof_cells, dloads)
    for dofidx in 1:length(f)
        r = dof_cells.offsets[dofidx] : dof_cells.offsets[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells.values[i]
            f[dofidx] += dloads[cellidx][localidx]
        end
    end
    return
end

#=
function update_f!(f::CuVector, dof_cells, dloads)
    args = (f, dof_cells.offsets, dof_cells.values, dloads)
    callkernel(dev, assemble_kernel2, args)
    CUDAdrv.synchronize(ctx)

    return
end

function assemble_kernel2(f, dof_cells_offsets, dof_cells_values, dloads)
    i = @thread_global_index()
    offset = @total_threads()
    @inbounds while i <= length(f)
        r = dof_cells_offsets[i] : dof_cells_offsets[i+1]-1
        for i in r
            cellidx, localidx = dof_cells_values[i]
            f[i] += dloads[cellidx][localidx]
        end
        i += offset
    end
    return
end
=#
