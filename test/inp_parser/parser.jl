using TopOpt.TopOptProblems.InputOutput.INP
using Ferrite, Test

cube = INP.Parser.import_inp(joinpath(@__DIR__, "testcube.inp"))
dh = cube.dh
grid = dh.grid
cells = grid.cells
nodes = grid.nodes
@test Ferrite.getspatialdim(grid) == 3 # 3D grid
@test Ferrite.nfaces(cells[1]) == 4 # Tetrahedron
@test Ferrite.nfacets(cells[1]) == 4 # Facets in Ferrite 1.x
@test Ferrite.nnodes(cells[1]) == 10 # Quadratic tetrahedron
@test haskey(grid.facetsets, "DLOAD_SET_1")

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

# Test triangular elements (CPS3)
raw_inp = INP.Parser.extract_inp(joinpath(@__DIR__, "triangle.inp"))
@test raw_inp.celltype == "CPS3"
@test length(raw_inp.node_coords) == 42
@test length(raw_inp.cells) == 60
@test raw_inp.cells[1] == (1, 2, 8)
@test raw_inp.cells[60] == (35, 42, 41)
@test raw_inp.node_coords[1] == (0.0, 0.0)
@test raw_inp.node_coords[42] == (60.0, 50.0)
@test raw_inp.E == 210000.0
@test raw_inp.ν == 0.3
@test raw_inp.nodedbcs["fixed_support"] == [(1, 0.0), (2, 0.0)]
@test raw_inp.cloads[42] == [0.0, -1000.0]

# Test parsed triangular mesh
triangle = INP.Parser.import_inp(joinpath(@__DIR__, "triangle.inp"))
dh = triangle.dh
grid = dh.grid
@test length(grid.nodes) == 42
@test length(grid.cells) == 60
@test Ferrite.nnodes(grid.cells[1]) == 3  # Linear triangle
@test typeof(grid.cells[1]) <: Ferrite.AbstractCell  # 2D triangle cell
@test Ferrite.getrefshape(typeof(grid.cells[1])) == Ferrite.RefTriangle
@test Ferrite.nnodes(grid.cells[1]) == 3
@test Ferrite.nfacets(grid.cells[1]) == 3

# Test inpcelltype function - maps Ferrite cell types to INP cell type strings
@test INP.Parser.inpcelltype(Ferrite.Triangle) == "CPS3"
@test INP.Parser.inpcelltype(Ferrite.QuadraticTriangle) == "CPS6"
@test INP.Parser.inpcelltype(Ferrite.Tetrahedron) == "C3D4"
@test INP.Parser.inpcelltype(Ferrite.QuadraticTetrahedron) == "C3D10"
@test INP.Parser.inpcelltype(Ferrite.Quadrilateral) == "CPS4"
@test INP.Parser.inpcelltype(Ferrite.QuadraticQuadrilateral) == "CPS8"
@test INP.Parser.inpcelltype(Ferrite.Hexahedron) == "C3D8"
@test INP.Parser.inpcelltype(Ferrite.SerendipityQuadraticHexahedron) == "C3D20"
@test INP.Parser.inpcelltype(Int) == ""  # Unknown type returns empty string

# Test C3D20 serendipity quadratic hexahedron parsing and conversion
raw_c3d20 = INP.Parser.extract_inp(joinpath(@__DIR__, "c3d20cube.inp"))
@test raw_c3d20.celltype == "C3D20"
@test raw_c3d20.cells[1] ==
    (1, 3, 8, 6, 13, 15, 20, 18, 2, 5, 7, 4, 14, 17, 19, 16, 9, 10, 12, 11)
@test raw_c3d20.E == 1000.0
@test raw_c3d20.ν == 0.3
@test raw_c3d20.nodedbcs["FixedBase"] == [(1, 0.0), (2, 0.0), (3, 0.0)]
@test raw_c3d20.cloads[19] == [0.0, -1.0, 0.0]

# Test imported C3D20 grid matches Ferrite's native SerendipityQuadraticHexahedron
c3d20 = INP.Parser.import_inp(joinpath(@__DIR__, "c3d20cube.inp"))
dh_c3d20 = c3d20.dh
grid_c3d20 = dh_c3d20.grid
@test length(grid_c3d20.nodes) == 20
@test length(grid_c3d20.cells) == 1
@test typeof(grid_c3d20.cells[1]) <: Ferrite.SerendipityQuadraticHexahedron
@test Ferrite.nnodes(grid_c3d20.cells[1]) == 20
@test Ferrite.nfacets(grid_c3d20.cells[1]) == 6

# Build a reference grid directly from Ferrite and compare coordinates/connectivity
ref_grid = Ferrite.generate_grid(Ferrite.SerendipityQuadraticHexahedron, (1, 1, 1))
@test [n.x for n in grid_c3d20.nodes] == [n.x for n in ref_grid.nodes]
@test grid_c3d20.cells[1].nodes == ref_grid.cells[1].nodes
