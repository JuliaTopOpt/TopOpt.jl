module VTK

using ...TopOptProblems: TopOptProblems, StiffnessTopOptProblem, HeatTransferTopOptProblem, Ferrite
using WriteVTK

export save_mesh

function save_mesh(filename::AbstractString, problem::StiffnessTopOptProblem)
    topology = ones(Ferrite.getncells(TopOptProblems.getdh(problem).grid))
    vtkfile = WriteVTK.vtk_grid(filename, problem, topology)
    return outfiles = WriteVTK.vtk_save(vtkfile)
end
function save_mesh(filename::AbstractString, problem, solver)
    return save_mesh(filename, problem, solver.vars)
end
function save_mesh(filename::AbstractString, alg)
    problem = alg.obj.problem
    vars = alg.optimizer.obj.solver.vars
    return save_mesh(filename, problem, vars)
end
function save_mesh(filename::AbstractString, problem, vars::AbstractVector)
    vtkfile = WriteVTK.vtk_grid(filename, problem, vars)
    return outfiles = WriteVTK.vtk_save(vtkfile)
end
function WriteVTK.vtk_grid(
    filename::AbstractString,
    problem::StiffnessTopOptProblem{dim,T},
    ρ::AbstractVector{T},
) where {dim,T}
    grid = problem.ch.dh.grid
    nel = length(TopOptProblems.getdh(problem).grid.cells)

    # ρ should be a full density vector (length = nel)
    if length(ρ) != nel
        throw(ArgumentError("Density vector ρ must have length equal to number of cells ($nel)"))
    end

    celltype = Ferrite.cell_to_vtkcell(Ferrite.getcelltype(grid))
    cls = Ferrite.MeshCell[]
    for (i, cell) in enumerate(Ferrite.CellIterator(grid))
        if ρ[i] >= 0.5
            push!(cls, Ferrite.MeshCell(celltype, copy(Ferrite.getnodes(cell))))
        end
    end
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

# Support for HeatTransferTopOptProblem
function save_mesh(filename::AbstractString, problem::HeatTransferTopOptProblem)
    topology = ones(Ferrite.getncells(TopOptProblems.getdh(problem).grid))
    vtkfile = WriteVTK.vtk_grid(filename, problem, topology)
    return outfiles = WriteVTK.vtk_save(vtkfile)
end

function WriteVTK.vtk_grid(
    filename::AbstractString,
    problem::HeatTransferTopOptProblem{dim,T},
    ρ::AbstractVector{T},
) where {dim,T}
    grid = problem.ch.dh.grid
    nel = length(TopOptProblems.getdh(problem).grid.cells)

    # ρ should be a full density vector (length = nel)
    if length(ρ) != nel
        throw(ArgumentError("Density vector ρ must have length equal to number of cells ($nel)"))
    end

    celltype = Ferrite.cell_to_vtkcell(Ferrite.getcelltype(grid))
    cls = Ferrite.MeshCell[]
    for (i, cell) in enumerate(Ferrite.CellIterator(grid))
        if ρ[i] >= 0.5
            push!(cls, Ferrite.MeshCell(celltype, copy(Ferrite.getnodes(cell))))
        end
    end
    coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    return vtk_grid(filename, coords, cls)
end

end
