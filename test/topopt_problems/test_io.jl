using TopOpt
using TopOpt.TopOptProblems
using Test
using Ferrite
using LinearAlgebra
using JSON

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
