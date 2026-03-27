# Tests for RectilinearGrid and other grid types
using TopOpt.TopOptProblems
using TopOpt.TopOptProblems: RectilinearGrid, LGrid, TieBeamGrid, nnodespercell, nfacespercell, nnodes, left, right, bottom, top, middlex, middley, middlez, back, front
using Ferrite
using Test

# Tests for RectilinearGrid
@testset "RectilinearGrid Basic Construction" begin
    # 2D grid
    nels = (10, 5)
    sizes = (1.0, 1.0)
    grid = RectilinearGrid(Val{:Linear}, nels, sizes)
    
    @test grid.nels == nels
    @test grid.sizes == sizes
    @test length(grid.white_cells) == prod(nels)
    @test length(grid.black_cells) == prod(nels)
    @test length(grid.constant_cells) == prod(nels)
    
    # Test corners
    @test grid.corners[1] ≈ Ferrite.Vec{2}((0.0, 0.0))
    @test grid.corners[2] ≈ Ferrite.Vec{2}((10.0, 5.0))
    
    # 3D grid
    nels3d = (4, 3, 2)
    sizes3d = (0.5, 0.5, 0.5)
    grid3d = RectilinearGrid(Val{:Linear}, nels3d, sizes3d)
    
    @test grid3d.nels == nels3d
    @test grid3d.sizes == sizes3d
    @test length(grid3d.white_cells) == prod(nels3d)
end

@testset "RectilinearGrid Linear vs Quadratic" begin
    # Linear 2D grid
    grid_linear = RectilinearGrid(Val{:Linear}, (6, 4), (1.0, 1.0))
    @test nnodespercell(grid_linear) == 4  # Quadrilateral
    @test nfacespercell(grid_linear) == 4
    
    # Quadratic 2D grid
    grid_quad = RectilinearGrid(Val{:Quadratic}, (6, 4), (1.0, 1.0))
    @test nnodespercell(grid_quad) == 9  # QuadraticQuadrilateral
    @test nfacespercell(grid_quad) == 4
    
    # 3D Linear grid (Hexahedron)
    grid_3d = RectilinearGrid(Val{:Linear}, (4, 3, 2), (1.0, 1.0, 1.0))
    @test nnodespercell(grid_3d) == 8  # Hexahedron
    @test nfacespercell(grid_3d) == 6
end

@testset "RectilinearGrid Position Methods" begin
    grid = RectilinearGrid(Val{:Linear}, (10, 5), (1.0, 2.0))
    
    # left: x[1] ≈ corners[1][1]
    @test left(grid, Ferrite.Vec{2}((0.0, 3.0))) == true
    @test left(grid, Ferrite.Vec{2}((0.5, 3.0))) == false
    
    # right: x[1] ≈ corners[2][1]
    @test right(grid, Ferrite.Vec{2}((10.0, 3.0))) == true
    @test right(grid, Ferrite.Vec{2}((9.5, 3.0))) == false
    
    # bottom: x[2] ≈ corners[1][2]
    @test bottom(grid, Ferrite.Vec{2}((3.0, 0.0))) == true
    @test bottom(grid, Ferrite.Vec{2}((3.0, 1.0))) == false
    
    # top: x[2] ≈ corners[2][2]
    @test top(grid, Ferrite.Vec{2}((3.0, 10.0))) == true
    @test top(grid, Ferrite.Vec{2}((3.0, 9.0))) == false
    
    # middlex: x[1] ≈ (corners[1][1] + corners[2][1]) / 2
    @test middlex(grid, Ferrite.Vec{2}((5.0, 3.0))) == true
    @test middlex(grid, Ferrite.Vec{2}((5.5, 3.0))) == false
    
    # middley: x[2] ≈ (corners[1][2] + corners[2][2]) / 2
    @test middley(grid, Ferrite.Vec{2}((3.0, 5.0))) == true
    @test middley(grid, Ferrite.Vec{2}((3.0, 5.5))) == false
end

@testset "RectilinearGrid 3D Position Methods" begin
    grid3d = RectilinearGrid(Val{:Linear}, (10, 5, 4), (1.0, 1.0, 0.5))
    
    # back: x[3] ≈ corners[1][3]
    @test back(grid3d, Ferrite.Vec{3}((3.0, 2.0, 0.0))) == true
    @test back(grid3d, Ferrite.Vec{3}((3.0, 2.0, 0.5))) == false
    
    # front: x[3] ≈ corners[2][3]
    @test front(grid3d, Ferrite.Vec{3}((3.0, 2.0, 2.0))) == true
    @test front(grid3d, Ferrite.Vec{3}((3.0, 2.0, 1.5))) == false
    
    # middlez: x[3] ≈ (corners[1][3] + corners[2][3]) / 2
    @test middlez(grid3d, Ferrite.Vec{3}((3.0, 2.0, 1.0))) == true
    @test middlez(grid3d, Ferrite.Vec{3}((3.0, 2.0, 1.1))) == false
end

@testset "RectilinearGrid Cell Properties" begin
    # Test nnodespercell and nfacespercell
    grid2d = RectilinearGrid(Val{:Linear}, (6, 4), (1.0, 1.0))
    @test nnodespercell(grid2d) == 4  # Quadrilateral
    @test nfacespercell(grid2d) == 4
    
    grid3d = RectilinearGrid(Val{:Linear}, (4, 3, 2), (1.0, 1.0, 1.0))
    @test nnodespercell(grid3d) == 8  # Hexahedron
    @test nfacespercell(grid3d) == 6
    
    # Test nnodes on Ferrite cell types
    quad = Ferrite.Quadrilateral((1, 2, 3, 4))
    @test TopOptProblems.nnodes(quad) == 4
    @test TopOptProblems.nnodes(typeof(quad)) == 4
    
    hex = Ferrite.Hexahedron((1, 2, 3, 4, 5, 6, 7, 8))
    @test TopOptProblems.nnodes(hex) == 8
    @test TopOptProblems.nnodes(typeof(hex)) == 8
end

@testset "LGrid Construction" begin
    # Default LGrid with keyword arguments
    lgrid = LGrid(Val{:Linear}, Float64, upperslab=30, lowerslab=70)
    @test lgrid isa Ferrite.Grid
    
    # Custom LGrid with explicit parameters
    LL = Ferrite.Vec{2,Float64}((0.0, 0.0))
    UR = Ferrite.Vec{2,Float64}((2.0, 4.0))
    MR = Ferrite.Vec{2,Float64}((4.0, 2.0))
    
    # Linear LGrid
    lgrid_linear = LGrid(Val{:Linear}, (2, 4), (2, 2), LL, UR, MR)
    @test lgrid_linear isa Ferrite.Grid
    
    # Quadratic LGrid
    lgrid_quad = LGrid(Val{:Quadratic}, (2, 4), (2, 2), LL, UR, MR)
    @test lgrid_quad isa Ferrite.Grid
end

@testset "TieBeamGrid Construction" begin
    # Linear TieBeamGrid
    tb_linear = TieBeamGrid(Val{:Linear}, Float64, 1)
    @test tb_linear isa Ferrite.Grid
    
    # Quadratic TieBeamGrid
    tb_quad = TieBeamGrid(Val{:Quadratic}, Float64, 1)
    @test tb_quad isa Ferrite.Grid
    
    # Default type parameter (no refine, defaults to 1)
    tb_default = TieBeamGrid(Val{:Linear})
    @test tb_default isa Ferrite.Grid
end

@testset "Grid Boundary Conditions" begin
    lgrid = LGrid(Val{:Linear}, Float64, upperslab=30, lowerslab=70)
    
    # Check that expected face sets exist
    @test haskey(lgrid.facesets, "right")
    @test haskey(lgrid.facesets, "top")
    
    # Check that load nodeset exists
    @test haskey(lgrid.nodesets, "load")
    
    tbgrid = TieBeamGrid(Val{:Linear}, Float64, 1)
    @test haskey(tbgrid.facesets, "leftfixed")
    @test haskey(tbgrid.facesets, "toproller")
    @test haskey(tbgrid.facesets, "rightload")
    @test haskey(tbgrid.facesets, "bottomload")
end