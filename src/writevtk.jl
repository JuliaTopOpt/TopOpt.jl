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

    celltype = JuAFEM.cell_to_vtkcell(JuAFEM.getcelltype(grid))
    cls = JuAFEM.MeshCell[]
    for (i, cell) in enumerate(CellIterator(grid))
        if black[i]
            push!(cls, JuAFEM.MeshCell(celltype, copy(getnodes(cell))))
        elseif !white[i]
            if vars[varind[i]] >= 0.5
                push!(cls, JuAFEM.MeshCell(celltype, copy(getnodes(cell))))
            end
        end
    end
    coords = reinterpret(T, getnodes(grid), (dim, getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end
