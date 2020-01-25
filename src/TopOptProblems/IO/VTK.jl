module VTK

using ...TopOptProblems: TopOptProblems, StiffnessTopOptProblem, JuAFEM
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

    celltype = JuAFEM.cell_to_vtkcell(JuAFEM.getcelltype(grid))
    cls = JuAFEM.MeshCell[]
    for (i, cell) in enumerate(JuAFEM.CellIterator(grid))
        if full_top
            if vars[i] >= 0.5
                push!(cls, JuAFEM.MeshCell(celltype, copy(JuAFEM.getnodes(cell))))
            end
        else
            if black[i]
                push!(cls, JuAFEM.MeshCell(celltype, copy(JuAFEM.getnodes(cell))))
            elseif !white[i]
                if vars[varind[i]] >= 0.5
                    push!(cls, JuAFEM.MeshCell(celltype, copy(JuAFEM.getnodes(cell))))
                end
            end
        end
    end
    coords = reshape(reinterpret(T, JuAFEM.getnodes(grid)), (dim, JuAFEM.getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

end
