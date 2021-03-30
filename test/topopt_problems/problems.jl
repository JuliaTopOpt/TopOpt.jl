using TopOpt.TopOptProblems
using Test

using Ferrite
using TopOpt.TopOptProblems: boundingbox

E = 1.0
ν = 0.3
force = 1.0

# Cantilever beam problem tests
@testset "Point load cantilever beam" begin
    global E, ν, force
    problem = PointLoadCantilever(Val{:Linear}, (160,40), (1.0,1.0), E, ν, force);
    ncells = 160*40
    @test problem.E == E
    @test problem.ν == ν
    @test problem.black == problem.white == falses(ncells)
    @test problem.force == force
    @test problem.force_dof == 161 * 21 * 2
    @test problem.varind == 1:ncells
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ problem.rect_grid.corners[i][j] atol = 1e-8
    end
    @test length(grid.boundary_matrix.nzval) == 2*160 + 2*40
    for (c, f) in grid.facesets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1,2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2,3]]
            @test grid.nodes[n].x[1] ≈ 160 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3,4]]
            @test grid.nodes[n].x[2] ≈ 40 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1,4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["bottom", "right", "top", "left"])) == 2*160 + 2*40
end

# Half MBB beam problem
@testset "Half MBB beam" begin
    global E, ν, force
    problem = HalfMBB(Val{:Linear}, (60,20), (1.,1.), E, ν, force); 
    ncells = 60*20
    @test problem.E == E
    @test problem.ν == ν
    @test problem.black == problem.white == falses(ncells)
    @test problem.force == force
    @test problem.force_dof == (61 * 20 + 2) * 2
    @test problem.varind == 1:ncells
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ problem.rect_grid.corners[i][j] atol = 1e-8
    end
    @test length(grid.boundary_matrix.nzval) == 2*60 + 2*20
    for (c, f) in grid.facesets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1,2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2,3]]
            @test grid.nodes[n].x[1] ≈ 60 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3,4]]
            @test grid.nodes[n].x[2] ≈ 20 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1,4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["bottom", "right", "top", "left"])) == 2*60 + 2*20
end

# L-beam problem
@testset "L-beam" begin
    global E, ν, force
    problem = LBeam(Val{:Linear}, Float64, force=force); 
    ncells = 100*50 + 50*50
    @test problem.E == E
    @test problem.ν == ν
    @test problem.black == problem.white == falses(ncells)
    @test problem.force == force
    @test problem.force_dof == (51 * 51 + 50 * 26) * 2
    @test problem.varind == 1:ncells
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    corners = [[0.0, 0.0], [100.0, 100.0]]
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ corners[i][j] atol = 1e-8
    end
    @test length(grid.boundary_matrix.nzval) == 100 * 2 + 50 * 4
    for (c, f) in grid.facesets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2,3]]
            @test grid.nodes[n].x[1] ≈ 100 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3,4]]
            @test grid.nodes[n].x[2] ≈ 100 atol = 1e-8
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["right", "top"])) == 2*50
end

# Tie-beam problem
@testset "Tie-beam" begin
    problem = TopOptProblems.TieBeam(Val{:Quadratic}, Float64);
    ncells = 100
    @test problem.E == 1
    @test problem.ν == 0.3
    @test problem.black == problem.white == falses(ncells)
    @test problem.force == 1
    @test problem.varind == 1:ncells
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    corners = [[0.0, 0.0], [32.0, 7.0]]
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ corners[i][j] atol = 1e-8
    end
    @test length(grid.boundary_matrix.nzval) == 32 * 2 + 3 * 2 + 4 * 2
    for (c, f) in grid.facesets["bottomload"]
        @test f == 1
        for n in grid.cells[c].nodes[[1,2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["rightload"]
        @test f == 2
        for n in grid.cells[c].nodes[[2,3]]
            @test grid.nodes[n].x[1] ≈ 32 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["toproller"]
        @test f == 3
        for n in grid.cells[c].nodes[[3,4]]
            @test grid.nodes[n].x[2] ≈ 7 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["leftfixed"]
        @test f == 4
        for n in grid.cells[c].nodes[[1,4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["bottomload", "rightload", "toproller", "leftfixed"])) == 8
	@test Ferrite.getorder(problem.ch.dh.field_interpolations[1]) == 2
	@test Ferrite.nnodes(grid.cells[1]) == 9
end
