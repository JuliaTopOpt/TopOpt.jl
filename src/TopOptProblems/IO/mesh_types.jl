using GeometryBasics: GeometryBasics
using GeometryTypes: GeometryTypes
using ..TopOptProblems:
    getdh, AbstractTopOptProblem, StiffnessTopOptProblem, QuadraticHexahedron
using Ferrite, VTKDataTypes

"""
map Ferrite cell type to VTKDataTypes cell type
"""
const ferrite_to_vtk = Dict(
    Triangle => 5,
    QuadraticTriangle => 22,
    Quadrilateral => 9,
    QuadraticQuadrilateral => 23,
    Tetrahedron => 10,
    QuadraticTetrahedron => 24,
    Hexahedron => 12,
    QuadraticHexahedron => 25,
)

"""
Converting a Ferrite grid to a VTKUnstructuredData from [VTKDataTypes](https://github.com/mohamed82008/VTKDataTypes.jl).
"""
function VTKDataTypes.VTKUnstructuredData(
    grid::Ferrite.Grid{dim,<:Ferrite.Cell{dim,N,M},T}
) where {dim,N,M,T}
    celltype = ferrite_to_vtk[eltype(grid.cells)]
    celltypes = [celltype for i in 1:length(grid.cells)]
    connectivity = copy(reinterpret(NTuple{N,Int}, grid.cells))
    node_coords = copy(reshape(reinterpret(Float64, grid.nodes), dim, length(grid.nodes)))
    return VTKUnstructuredData(node_coords, celltypes, connectivity)
end
function VTKDataTypes.VTKUnstructuredData(problem::AbstractTopOptProblem)
    return VTKUnstructuredData(getdh(problem).grid)
end
function VTKDataTypes.GLMesh(grid::Ferrite.Grid; kwargs...)
    return GLMesh(VTKUnstructuredData(grid); kwargs...)
end
function VTKDataTypes.GLMesh(problem::AbstractTopOptProblem; kwargs...)
    return GLMesh(VTKUnstructuredData(problem); kwargs...)
end

```
workaround taken from https://github.com/JuliaPlots/Makie.jl/issues/647
TODO: should directly convert VTKDataTypes.VTKUnstructuredData to GeometryBasics.Mesh
Do not want to spend more time on this now...
```
function GeometryBasics.Mesh(glmesh::GeometryTypes.GLNormalVertexcolorMesh)
    newverts = reinterpret(GeometryBasics.Point{3,Float32}, glmesh.vertices)
    newfaces = reinterpret(
        GeometryBasics.NgonFace{3,GeometryBasics.OffsetInteger{-1,UInt32}}, glmesh.faces
    )
    newnormals = reinterpret(GeometryBasics.Vec{3,Float32}, glmesh.normals)
    return GeometryBasics.Mesh(
        GeometryBasics.meta(newverts; normals=newnormals, color=glmesh.color), newfaces
    )
end

"""
Get mesh of the topopt problem with a given topology indicator vector
"""
function GeometryBasics.Mesh(
    problem::AbstractTopOptProblem, topology::Array{T,1}; kwargs...
) where {T}
    mesh = VTKUnstructuredData(problem)
    topology = round.(topology)
    inds = findall(isequal(0), topology)
    deleteat!(mesh.cell_connectivity, inds)
    deleteat!(mesh.cell_types, inds)
    topology = topology[setdiff(1:length(topology), inds)]
    mesh.cell_data["topology"] = topology
    glmesh = GLMesh(mesh; color="topology", kwargs...)
    return GeometryBasics.Mesh(glmesh)
end
