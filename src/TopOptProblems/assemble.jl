function assemble(
    problem::AbstractTopOptProblem,
    elementinfo::ElementFEAInfo,
    vars=ones(floattype(problem), getncells(getdh(problem).grid)),
    penalty=PowerPenalty(floattype(problem)(3.0)),
    xmin=floattype(problem)(0.001),
)
    T = floattype(problem)
    dim = getdim(problem)
    globalinfo = GlobalFEAInfo(problem)
    assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)
    return globalinfo
end

# Assembly for all problem types
# For structural: fes contains body forces (penalized), fixedload contains concentrated/distributed loads (not penalized)
# For heat transfer: fes is zeros (no body forces), fixedload contains heat source (not penalized)
# 
# Note: ρ should be a full density vector (length = nel) with values already accounting for 
# black (density=1) and white (density=xmin) elements. Use FixedElementProjector to map 
# free variables to full densities.
function assemble!(
    globalinfo::GlobalFEAInfo,
    problem::AbstractTopOptProblem,
    elementinfo::ElementFEAInfo,
    ρ=ones(floattype(problem), getncells(getdh(problem).grid)),
    penalty=PowerPenalty(floattype(problem)(3.0)),
    xmin=floattype(problem)(0.001);
    assemble_f=true,
)
    T = floattype(problem)
    dim = getdim(problem)
    ch = problem.ch
    dh = ch.dh
    K, f = globalinfo.K, globalinfo.f
    if assemble_f
        f .= elementinfo.fixedload
    end
    Kes, fes = elementinfo.Kes, elementinfo.fes

    _K = K isa Symmetric ? K.data : K
    _K.nzval .= 0
    assembler = Ferrite.AssemblerSparsityPattern(_K, f, Int[], Int[])

    global_dofs = zeros(Int, ndofs_per_cell(dh))
    fe = zeros(typeof(fes[1]))
    Ke = zeros(T, size(rawmatrix(Kes[1])))

    celliterator = CellIterator(dh)
    for (i, cell) in enumerate(celliterator)
        # get global_dofs for cell#i
        celldofs!(global_dofs, dh, i)
        fe = fes[i]
        _Ke = rawmatrix(Kes[i])
        Ke = _Ke isa Symmetric ? _Ke.data : _Ke
        
        # Apply density interpolation
        if PENALTY_BEFORE_INTERPOLATION
            px = density(penalty(ρ[i]), xmin)
        else
            px = penalty(density(ρ[i], xmin))
        end
        Ke = px * Ke
        if assemble_f
            fe = px * fe
            Ferrite.assemble!(assembler, global_dofs, Ke, fe)
        else
            Ferrite.assemble!(assembler, global_dofs, Ke)
        end
    end

    #* apply boundary condition
    TK = eltype(K)
    _K = TK <: Symmetric ? K.data : K
    apply!(_K, f, ch)

    return nothing
end

function assemble_f(
    problem::StiffnessTopOptProblem{dim,T},
    elementinfo::ElementFEAInfo{dim,T},
    vars::AbstractVector{T},
    penalty,
    xmin=T(1) / 1000,
) where {dim,T}
    f = get_f(problem, vars)
    assemble_f!(f, problem, elementinfo, vars, penalty, xmin)
    return f
end
get_f(problem, vars::Array) = zeros(floattype(problem), ndofs(problem.ch.dh))

function assemble_f!(
    f::AbstractVector,
    problem::StiffnessTopOptProblem,
    elementinfo::ElementFEAInfo,
    ρ::AbstractVector,
    penalty,
    xmin,
)
    fes = elementinfo.fes

    dof_cells = elementinfo.metadata.dof_cells

    update_f!(
        f, fes, elementinfo.fixedload, dof_cells, ρ, penalty, xmin
    )
    return f
end

function update_f!(
    f::Vector, fes, fixedload, dof_cells, ρ, penalty, xmin
)
    @inbounds for dofidx in 1:length(f)
        f[dofidx] = fixedload[dofidx]
        r = dof_cells.offsets[dofidx]:(dof_cells.offsets[dofidx + 1] - 1)
        for i in r
            cellidx, localidx = dof_cells.values[i]
            if PENALTY_BEFORE_INTERPOLATION
                px = density(penalty(ρ[cellidx]), xmin)
            else
                px = penalty(density(ρ[cellidx], xmin))
            end
            f[dofidx] += px * fes[cellidx][localidx]
        end
    end

    return nothing
end

function assemble_f!(f::AbstractVector, problem, dloads)
    metadata = problem.metadata
    dof_cells = metadata.dof_cells
    update_f!(f, dof_cells, dloads)
    return f
end

function update_f!(f::Vector, dof_cells, dloads)
    for dofidx in 1:length(f)
        r = dof_cells.offsets[dofidx]:(dof_cells.offsets[dofidx + 1] - 1)
        for i in r
            cellidx, localidx = dof_cells.values[i]
            f[dofidx] += dloads[cellidx][localidx]
        end
    end
    return nothing
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
