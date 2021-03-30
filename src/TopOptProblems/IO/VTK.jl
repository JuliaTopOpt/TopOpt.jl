module VTK

using ...TopOptProblems: TopOptProblems, StiffnessTopOptProblem, Ferrite
using WriteVTK

export  save_mesh

function save_mesh(filename, problem::StiffnessTopOptProblem)
    topology = ones(getncells(TopOptProblems.getdh(problem).grid))
    vtkfile = WriteVTK.vtk_grid(filename, problem, topology)
    outfiles = WriteVTK.vtk_save(vtkfile)
end
function save_mesh(filename, problem, solver)
    save_mesh(filename, problem, solver.vars)
end
function save_mesh(filename, alg)
    problem = alg.obj.problem
    vars = alg.optimizer.obj.solver.vars
    save_mesh(filename, problem, vars)
end
function save_mesh(filename, problem, vars::AbstractVector)
    vtkfile = WriteVTK.vtk_grid(filename, problem, vars)
    outfiles = WriteVTK.vtk_save(vtkfile)
end
function WriteVTK.vtk_grid(filename::AbstractString, problem::StiffnessTopOptProblem{dim, T}, vars::AbstractVector{T}) where {dim, T}
    varind = problem.varind
    black = problem.black
    white = problem.white
    grid = problem.ch.dh.grid
    full_top = length(vars) == length(TopOptProblems.getdh(problem).grid.cells)

    celltype = Ferrite.cell_to_vtkcell(Ferrite.getcelltype(grid))
    cls = Ferrite.MeshCell[]
    for (i, cell) in enumerate(Ferrite.CellIterator(grid))
        if full_top
            if vars[i] >= 0.5
                push!(cls, Ferrite.MeshCell(celltype, copy(Ferrite.getnodes(cell))))
            end
        else
            if black[i]
                push!(cls, Ferrite.MeshCell(celltype, copy(Ferrite.getnodes(cell))))
            elseif !white[i]
                if vars[varind[i]] >= 0.5
                    push!(cls, Ferrite.MeshCell(celltype, copy(Ferrite.getnodes(cell))))
                end
            end
        end
    end
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

end
