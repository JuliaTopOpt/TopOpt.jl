using TopOpt
using TopOpt.TopOptProblems
using Test
using Ferrite
using LinearAlgebra

@testset "VTK I/O" begin
    # Create a simple problem
    problem = PointLoadCantilever(Val{:Linear}, (10, 6), (1.0, 1.0), 1.0, 0.3, 1.0)

    # Test save_mesh function exists
    @test isdefined(TopOpt.TopOptProblems.InputOutput, :save_mesh)

    # Create temporary file for VTK output
    mktempdir() do tmpdir
        vtk_path = joinpath(tmpdir, "test_output")
        # Test VTK saving with default densities
        @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(vtk_path, problem)
        # Verify files were created - check for .vtu or .pvtu files
        vtk_files = filter(f -> occursin("test_output", f) && (endswith(f, ".vtu") || endswith(f, ".pvtu")), readdir(tmpdir))
        @test length(vtk_files) > 0
        # Test with custom densities
        densities = fill(0.5, getncells(problem.ch.dh.grid))
        @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(vtk_path * "_custom", problem, densities)
    end
end

@testset "Mesh types" begin
    # Test grid dimensions and properties - Ferrite grids are the mesh type
    problem = PointLoadCantilever(Val{:Linear}, (10, 6), (1.0, 1.0), 1.0, 0.3, 1.0)
    grid = problem.ch.dh.grid
    @test getncells(grid) == 60
    @test length(grid.nodes) == 77
    # Verify it's a Ferrite.Grid
    @test grid isa Ferrite.Grid
end

@testset "INP Parser" begin
    # Test INP file loading if files exist
    inp_dir = joinpath(@__DIR__, "..", "inp_parser")
    mbb_file = joinpath(inp_dir, "MBB.inp")
    @testset "INP file loading" begin
        # Test that we can read the INP file
        content = read(mbb_file, String)
        @test length(content) > 0
        @test occursin("*Part", content) || occursin("*Node", content)
    end
    testcube_file = joinpath(inp_dir, "testcube.inp")
    @testset "Test cube INP" begin
        content = read(testcube_file, String)
        @test length(content) > 0
    end
end

@testset "save_mesh comprehensive" begin
    mktempdir() do tmpdir
        @testset "Standard problem VTK export" begin
            # 2D problem
            problem = PointLoadCantilever(Val{:Linear}, (4, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
            vtk_path = joinpath(tmpdir, "test_2d")
            densities = fill(0.7, getncells(problem.ch.dh.grid))

            # Test save_mesh with densities
            @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(
                vtk_path, problem, densities
            )

            # Check files were created
            files = readdir(tmpdir)
            @test any(f -> occursin("test_2d", f) && (endswith(f, ".vtu") || endswith(f, ".pvtu")), files)
        end

        @testset "3D problem VTK export" begin
            # Skip if 3D problems not available or too slow
            problem_3d = PointLoadCantilever(Val{:Linear}, (4, 4, 2), (1.0, 1.0, 1.0), 1.0, 0.3, 1.0)
            vtk_path = joinpath(tmpdir, "test_3d")

            @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(vtk_path, problem_3d)

            files = readdir(tmpdir)
            @test any(f -> occursin("test_3d", f), files)
        end

        @testset "Heat problem VTK export" begin
            # Create a heat conduction problem
            nels = (4, 4)
            sizes = (1.0, 1.0)
            problem = HeatConductionProblem(Val{:Linear}, nels, sizes, 1.0; Tleft=0.0, Tright=0.0)
            vtk_path = joinpath(tmpdir, "test_heat")

            # Heat problems should also be exportable
            @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(vtk_path, problem)

            # Test with custom densities
            densities = fill(0.6, getncells(problem.ch.dh.grid))
            vtk_path2 = joinpath(tmpdir, "test_heat_densities")
            @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(vtk_path2, problem, densities)

            # Verify files were created
            files = readdir(tmpdir)
            @test any(f -> occursin("test_heat", f), files)
        end

        @testset "VTK with cell data" begin
            problem = PointLoadCantilever(Val{:Linear}, (4, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
            vtk_path = joinpath(tmpdir, "test_with_data")
            densities = rand(Float64, getncells(problem.ch.dh.grid))

            # Save with density data
            @test_nowarn TopOpt.TopOptProblems.InputOutput.save_mesh(
                vtk_path, problem, densities
            )
        end
    end
end

