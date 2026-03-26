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

@testset "JSON I/O utilities" begin
    # Test basic JSON handling for TopOpt
    test_data = Dict(
        "grid_size" => [10, 5],
        "E" => 1.0,
        "nu" => 0.3,
        "force" => 1.0
    )
    mktempdir() do tmpdir
        json_path = joinpath(tmpdir, "test_config.json")
        open(json_path, "w") do io
            JSON.print(io, test_data)
        end
        # Read back and verify
        loaded_data = JSON.parsefile(json_path)
        @test loaded_data["E"] == 1.0
        @test loaded_data["nu"] == 0.3
        @test loaded_data["grid_size"] == [10, 5]
    end
end

@testset "Problem I/O integration" begin
    # Test saving and loading problem data
    nels = (10, 6)
    sizes = (1.0, 1.0)
    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
    mktempdir() do tmpdir
        # Save problem state
        state_path = joinpath(tmpdir, "problem_state.json")
        # Extract key problem properties - access fields directly since getE/getν not exported
        state = Dict(
            "nels" => nels,
            "sizes" => sizes,
            "E" => problem.E,
            "nu" => problem.ν,
            "ncells" => getncells(problem),
            "ndofs" => ndofs(problem.ch.dh)
        )
        open(state_path, "w") do io
            JSON.print(io, state)
        end
        # Verify state file
        @test isfile(state_path)
        # Load and verify
        loaded_state = JSON.parsefile(state_path)
        @test loaded_state["E"] == 1.0
        @test loaded_state["nu"] == 0.3
        @test loaded_state["ncells"] == prod(nels)
    end
end