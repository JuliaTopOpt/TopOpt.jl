using TopOpt, Ferrite, Test

@testset "Ferrite 1.x upgrade behavior" begin
    @testset "2D PointLoadCantilever construction" begin
        problem = PointLoadCantilever(Val{:Linear}, (10, 6), (1.0, 1.0), 1.0, 0.3, 1.0)
        dh = getdh(problem)
        grid = dh.grid
        @test Ferrite.getspatialdim(grid) == 2
        @test typeof(grid.cells[1]) <: Ferrite.Quadrilateral
        @test Ferrite.nnodes(grid.cells[1]) == 4
        @test Ferrite.nfacets(grid.cells[1]) == 4
        meta = problem.metadata
        @test size(meta.cell_dofs, 1) == 8
        @test size(meta.node_dofs) == (2, Ferrite.getnnodes(grid))
        @test problem.force_dof in meta.node_dofs
    end

    @testset "3D PointLoadCantilever construction" begin
        problem3 = PointLoadCantilever(
            Val{:Linear}, (4, 4, 2), (1.0, 1.0, 1.0), 1.0, 0.3, 1.0
        )
        dh3 = getdh(problem3)
        grid3 = dh3.grid
        @test Ferrite.getspatialdim(grid3) == 3
        @test typeof(grid3.cells[1]) <: Ferrite.Hexahedron
        @test Ferrite.nnodes(grid3.cells[1]) == 8
        @test Ferrite.nfacets(grid3.cells[1]) == 6
    end

    @testset "2D HalfMBB construction" begin
        problem_mbb = HalfMBB(Val{:Linear}, (10, 6), (1.0, 1.0), 1.0, 0.3, 1.0)
        @test Ferrite.getspatialdim(getdh(problem_mbb).grid) == 2
    end

    @testset "C3D20 INP import matches native Ferrite grid" begin
        using TopOpt.TopOptProblems.InputOutput.INP
        c3d20 = INP.Parser.import_inp(joinpath(@__DIR__, "inp_parser/c3d20cube.inp"))
        dh_c3d20 = c3d20.dh
        grid_c3d20 = dh_c3d20.grid
        @test Ferrite.getspatialdim(grid_c3d20) == 3
        @test typeof(grid_c3d20.cells[1]) <: Ferrite.SerendipityQuadraticHexahedron
        @test Ferrite.nnodes(grid_c3d20.cells[1]) == 20
        @test Ferrite.nfacets(grid_c3d20.cells[1]) == 6

        ref_grid = Ferrite.generate_grid(Ferrite.SerendipityQuadraticHexahedron, (1, 1, 1))
        @test [n.x for n in grid_c3d20.nodes] == [n.x for n in ref_grid.nodes]
        @test grid_c3d20.cells[1].nodes == ref_grid.cells[1].nodes
    end
end