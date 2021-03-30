using ..CUDASupport
using ..TopOpt: @init_cuda
@init_cuda()
using ..GPUUtils
import ..TopOpt: whichdevice

get_f(problem, vars::CuArray) = f = zeros(typeof(vars), ndofs(problem.ch.dh))

function update_f!(f::CuVector{T}, fes, fixedload, dof_cells, black, 
    white, penalty, vars, varind, xmin) where {T}

    args = (f, fes, fixedload, dof_cells.offsets, dof_cells.values, black, 
        white, penalty, vars, varind, xmin, length(f))
    callkernel(dev, assemble_kernel1, args)
    CUDAdrv.synchronize(ctx)
end

function assemble_kernel1(f, fes, fixedload, dof_cells_offsets, dof_cells_values, black, 
    white, penalty, vars, varind, xmin, ndofs)

    dofidx = @thread_global_index()
    offset = @total_threads()

    while dofidx <= ndofs
        f[dofidx] = fixedload[dofidx]
        r = dof_cells_offsets[dofidx] : dof_cells_offsets[dofidx+1]-1
        for i in r
            cellidx, localidx = dof_cells_values[i]
            if black[cellidx]
                f[dofidx] += fes[cellidx][localidx]
            elseif white[cellidx]
                px = xmin
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
        dofidx += offset
    end

    return
end

whichdevice(p::StiffnessTopOptProblem) = whichdevice(p.ch)
whichdevice(ch::ConstraintHandler) = whichdevice(ch.dh)
whichdevice(dh::DofHandler) = whichdevice(dh.grid)
whichdevice(g::Ferrite.Grid) = whichdevice(g.cells)

@define_cu(ElementFEAInfo, :Kes, :fes, :fixedload, :cellvolumes, :metadata, :black, :white, :varind, :cells)
@define_cu(TopOptProblems.Metadata, :cell_dofs, :dof_cells, :node_cells, :node_dofs)
@define_cu(Ferrite.ConstraintHandler, :values, :prescribed_dofs, :dh)
@define_cu(Ferrite.DofHandler, :grid)
@define_cu(Ferrite.Grid, :cells)
for T in (PointLoadCantilever, HalfMBB, LBeam, TieBeam, InpStiffness)
    @eval @define_cu($T, :ch, :black, :white, :varind)
end
