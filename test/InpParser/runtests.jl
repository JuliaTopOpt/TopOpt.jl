using InpParser, JuAFEM

using Test

cube = InpParser.import_inp("testcube.inp")
dh = cube.dh
grid = dh.grid
cells = grid.cells
nodes = grid.nodes
getdim(::JuAFEM.Cell{N}) where {N} = N

@test dh.field_dims == [3]
@test getdim(cells[1]) == 3 # 3D cells
@test JuAFEM.nfaces(cells[1]) == 4 # Tetrahedron
@test JuAFEM.nnodes(cells[1]) == 10 # Quadratic tetrahedron
@test length(grid.boundary_matrix.nzval) == 16

raw_inp = InpParser.extract_inp("testcube.inp")
@test raw_inp.celltype == "C3D10"
@test raw_inp.E == 70_000
@test raw_inp.ν == 0.3

@test raw_inp.cellsets["Eall"] == 1:5
@test raw_inp.cellsets["Evolumes"] == 1:5
@test raw_inp.cellsets["SolidMaterialSolid"] == 1:5
force_node = collect(keys(raw_inp.cloads))[1]
@test raw_inp.node_coords[force_node] == (10, 10, 10)
@test raw_inp.cloads[force_node] == [0, -1, 0]

@test raw_inp.facesets["DLOAD_SET_1"] == [(1,3), (5,2)]
@test raw_inp.dloads["DLOAD_SET_1"] == 1

@test raw_inp.nodedbcs["FemConstraintDisplacement"] == [(1, 0), (2, 0), (3, 0)]
for n in raw_inp.nodesets["FemConstraintDisplacement"]
    @test raw_inp.node_coords[n][3] == 0
end
