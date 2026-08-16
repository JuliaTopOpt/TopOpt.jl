# Tests for RectilinearGrid and other grid types
using TopOpt.TopOptProblems
using TopOpt.TopOptProblems:
    RectilinearGrid,
    RectilinearTopology,
    LGrid,
    TieBeamGrid,
    nnodespercell,
    nfacespercell,
    nnodes,
    left,
    right,
    bottom,
    top,
    middlex,
    middley,
    middlez,
    back,
    front
using Ferrite
using Test

# Tests for RectilinearGrid
@testset "RectilinearGrid Basic Construction" begin
    # Invalid cell types fail fast with a descriptive error
    @test_throws ArgumentError RectilinearGrid((4, 4), (1.0, 1.0); celltype=:Cubic)
    @test_throws ArgumentError LGrid((2, 4), (2, 2), Vec{2,Float64}((0.0, 0.0)), Vec{2,Float64}((2.0, 4.0)), Vec{2,Float64}((4.0, 2.0)); celltype=:Cubic)

    # 2D grid
    nels = (10, 5)
    sizes = (1.0, 1.0)
    grid = RectilinearGrid(nels, sizes)

    @test grid.nels == nels
    @test grid.sizes == sizes

    # Test corners
    @test grid.corners[1] ≈ Ferrite.Vec{2}((0.0, 0.0))
    @test grid.corners[2] ≈ Ferrite.Vec{2}((10.0, 5.0))

    # 3D grid
    nels3d = (4, 3, 2)
    sizes3d = (0.5, 0.5, 0.5)
    grid3d = RectilinearGrid(nels3d, sizes3d)

    @test grid3d.nels == nels3d
    @test grid3d.sizes == sizes3d
end

@testset "RectilinearGrid Linear vs Quadratic" begin
    # Linear 2D grid
    grid_linear = RectilinearGrid((6, 4), (1.0, 1.0))
    @test nnodespercell(grid_linear) == 4  # Quadrilateral
    @test nfacespercell(grid_linear) == 4

    # Quadratic 2D grid
    grid_quad = RectilinearGrid((6, 4), (1.0, 1.0); celltype=:Quadratic)
    @test nnodespercell(grid_quad) == 9  # QuadraticQuadrilateral
    @test nfacespercell(grid_quad) == 4

    # 3D Linear grid (Hexahedron)
    grid_3d = RectilinearGrid((4, 3, 2), (1.0, 1.0, 1.0))
    @test nnodespercell(grid_3d) == 8  # Hexahedron
    @test nfacespercell(grid_3d) == 6
end

@testset "RectilinearGrid Position Methods" begin
    grid = RectilinearGrid((10, 5), (1.0, 2.0))

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
    grid3d = RectilinearGrid((10, 5, 4), (1.0, 1.0, 0.5))

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
    grid2d = RectilinearGrid((6, 4), (1.0, 1.0))
    @test nnodespercell(grid2d) == 4  # Quadrilateral
    @test nfacespercell(grid2d) == 4

    grid3d = RectilinearGrid((4, 3, 2), (1.0, 1.0, 1.0))
    @test nnodespercell(grid3d) == 8  # Hexahedron
    @test nfacespercell(grid3d) == 6

    # Test nnodes on Ferrite cell types
    quad = Ferrite.Quadrilateral((1, 2, 3, 4))
    @test TopOpt.TopOptProblems.nnodes(quad) == 4
    @test TopOpt.TopOptProblems.nnodes(typeof(quad)) == 4

    hex = Ferrite.Hexahedron((1, 2, 3, 4, 5, 6, 7, 8))
    @test TopOpt.TopOptProblems.nnodes(hex) == 8
    @test TopOpt.TopOptProblems.nnodes(typeof(hex)) == 8
end

@testset "LGrid Construction" begin
    # Default LGrid with keyword arguments
    lgrid = LGrid(Float64; upperslab=30, lowerslab=70)
    @test lgrid isa Ferrite.Grid

    # Custom LGrid with explicit parameters
    LL = Ferrite.Vec{2,Float64}((0.0, 0.0))
    UR = Ferrite.Vec{2,Float64}((2.0, 4.0))
    MR = Ferrite.Vec{2,Float64}((4.0, 2.0))

    # Linear LGrid
    lgrid_linear = LGrid((2, 4), (2, 2), LL, UR, MR)
    @test lgrid_linear isa Ferrite.Grid

    # Quadratic LGrid
    lgrid_quad = LGrid((2, 4), (2, 2), LL, UR, MR; celltype=:Quadratic)
    @test lgrid_quad isa Ferrite.Grid

    # Distributed load: `load_width` nodes share the force equally
    lgrid_pt = LGrid(Float64; length=20, height=20, upperslab=10, lowerslab=10)
    @test length(getnodeset(lgrid_pt, "load")) == 1
    lgrid_dist = LGrid(
        Float64; length=20, height=20, upperslab=10, lowerslab=10, load_width=5
    )
    load_nodes = getnodeset(lgrid_dist, "load")
    @test length(load_nodes) == 5
    # All load nodes lie on the right edge of the lower arm
    @test all(n -> lgrid_dist.nodes[n].x[1] == 20.0, load_nodes)
    # Centered on the single-node default location
    @test sum(lgrid_dist.nodes[n].x[2] for n in load_nodes) / 5 ≈
        lgrid_pt.nodes[only(getnodeset(lgrid_pt, "load"))].x[2]

    # Force split conserves the resultant
    problem_pt = LBeam(; length=20, height=20, upperslab=10, lowerslab=10, force=2.0)
    problem_dist = LBeam(;
        length=20, height=20, upperslab=10, lowerslab=10, force=2.0, load_width=5
    )
    cload_pt = getcloaddict(problem_pt)
    cload_dist = getcloaddict(problem_dist)
    @test length(cload_pt) == 1
    @test length(cload_dist) == 5
    @test sum(v -> v[2], values(cload_dist)) ≈ -2.0
    @test sum(v -> v[2], values(cload_pt)) ≈ -2.0
    # Same resultant at the same center => identical total torque about the origin
    torque(cl) = sum(((n, f),) -> lgrid_dist.nodes[n].x[1] * f[2], collect(cl))
    @test torque(cload_dist) ≈ torque(cload_pt)

    @test_throws ArgumentError LGrid(
        Float64; length=20, height=20, upperslab=10, lowerslab=10, load_width=0
    )
end

@testset "TieBeamGrid Construction" begin
    # Linear TieBeamGrid
    tb_linear = TieBeamGrid(Float64; refine=1)
    @test tb_linear isa Ferrite.Grid

    # Quadratic TieBeamGrid
    tb_quad = TieBeamGrid(Float64; celltype=:Quadratic, refine=1)
    @test tb_quad isa Ferrite.Grid

    # Default type parameter (no refine, defaults to 1)
    tb_default = TieBeamGrid()
    @test tb_default isa Ferrite.Grid
end

@testset "Grid Boundary Conditions" begin
    lgrid = LGrid(Float64; upperslab=30, lowerslab=70)

    # Check that expected face sets exist
    @test haskey(lgrid.facetsets, "right")
    @test haskey(lgrid.facetsets, "top")

    # Check that load nodeset exists
    @test haskey(lgrid.nodesets, "load")

    tbgrid = TieBeamGrid(Float64; refine=1)
    @test haskey(tbgrid.facetsets, "leftfixed")
    @test haskey(tbgrid.facetsets, "toproller")
    @test haskey(tbgrid.facetsets, "rightload")
    @test haskey(tbgrid.facetsets, "bottomload")
end

@testset "RectilinearTopology" begin
    # Create a problem with known dimensions
    nels = (10, 6)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(nels, sizes, E, ν, force)

    # Test 1: Default topology (all ones)
    topology = RectilinearTopology(problem)
    @test size(topology) == reverse(nels)
    @test all(topology .== 1.0)

    # Test 2: Custom topology with zeros
    custom_topology = zeros(Float64, Ferrite.getncells(problem))
    topology2 = RectilinearTopology(problem, custom_topology)
    @test size(topology2) == reverse(nels)
    @test all(topology2 .== 0.0)

    # Test 3: Custom topology with mixed values
    mixed_topology = ones(Float64, Ferrite.getncells(problem))
    mixed_topology[1:div(end, 2)] .= 0.5
    topology3 = RectilinearTopology(problem, mixed_topology)
    @test size(topology3) == reverse(nels)
    @test topology3 isa AbstractMatrix
    @test any(topology3 .== 0.5)
    @test any(topology3 .== 1.0)
end
