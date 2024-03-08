using TopOpt.TopOptProblems.InputOutput.INP
using Ferrite, Test

cube = INP.Parser.import_inp(joinpath(@__DIR__, "testcube.inp"))
dh = cube.dh
grid = dh.grid
cells = grid.cells
nodes = grid.nodes
getdim(::Ferrite.Cell{N}) where {N} = N

@test dh.field_dims == [3]
@test getdim(cells[1]) == 3 # 3D cells
@test Ferrite.nfaces(cells[1]) == 4 # Tetrahedron
@test Ferrite.nnodes(cells[1]) == 10 # Quadratic tetrahedron
@test length(grid.boundary_matrix.nzval) == 16

raw_inp = INP.Parser.extract_inp(joinpath(@__DIR__, "testcube.inp"))
@test raw_inp.celltype == "C3D10"
@test raw_inp.E == 70_000
@test raw_inp.ν == 0.3

@test raw_inp.cellsets["Eall"] == 1:5
@test raw_inp.cellsets["Evolumes"] == 1:5
@test raw_inp.cellsets["SolidMaterialSolid"] == 1:5
force_node = collect(keys(raw_inp.cloads))[1]
@test raw_inp.node_coords[force_node] == (10, 10, 10)
@test raw_inp.cloads[force_node] == [0, -1, 0]

@test raw_inp.facesets["DLOAD_SET_1"] == [(1, 3), (5, 2)]
@test raw_inp.dloads["DLOAD_SET_1"] == 1

@test raw_inp.nodedbcs["FemConstraintDisplacement"] == [(1, 0), (2, 0), (3, 0)]
@test raw_inp.nodesets["FemConstraintDisplacement"] == [1, 3, 5, 7, 13, 14, 15, 16, 22]
for n in raw_inp.nodesets["FemConstraintDisplacement"]
    @test raw_inp.node_coords[n][3] == 0
end

raw_inp = INP.Parser.extract_inp(joinpath(@__DIR__, "MBB.inp"))
# element type
@test raw_inp.celltype == "CPS4"
# node coordinates
@test raw_inp.node_coords[1] == (0.0, 0.0)
@test raw_inp.node_coords[2] == (5.0, 0.0)
@test raw_inp.node_coords[450] == (195.0, 50.0)
@test raw_inp.node_coords[451] == (200.0, 50.0)
# cell connectivity
@test raw_inp.cells[1] == (1, 2, 43, 42)
@test raw_inp.cells[2] == (2, 3, 44, 43)
@test raw_inp.cells[399] == (408, 409, 450, 449)
@test raw_inp.cells[400] == (409, 410, 451, 450)
# Dirichlet boundary conditions
@test raw_inp.nodedbcs["fixed_support"] == [(1, 0.0), (2, 0.0)]
@test raw_inp.nodedbcs["roller_support"] == [(1, 0.0)]
# concentrated load
@test raw_inp.cloads[431] == [0.0, -3.0]
# material density
@test raw_inp.density == 0
# Young's modulus
@test raw_inp.E == 42000
# Poisson ratio
@test raw_inp.ν == 0.2
