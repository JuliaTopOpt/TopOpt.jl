# Load the package
using Test
using Serialization, GeometryBasics
import GeometryTypes

file_path = joinpath(@__DIR__, "pointload_cantilever.glmesh")
glmesh = deserialize(file_path)

# workaround here: https://github.com/JuliaPlots/Makie.jl/issues/647
function GeometryBasics.Mesh(glmesh::GeometryTypes.GLNormalVertexcolorMesh)
    newverts = reinterpret(GeometryBasics.Point{3, Float32}, glmesh.vertices)
    newfaces = reinterpret(GeometryBasics.NgonFace{3, GeometryBasics.OffsetInteger{-1, UInt32}}, glmesh.faces)
    newnormals = reinterpret(GeometryBasics.Vec{3, Float32}, glmesh.normals)
    return GeometryBasics.Mesh(meta(newverts; normals = newnormals, color = glmesh.color), newfaces)
end

geo_basic_mesh = GeometryBasics.Mesh(glmesh)
@test isa(geo_basic_mesh, GeometryBasics.Mesh)
# mesh(geo_basic_mesh)