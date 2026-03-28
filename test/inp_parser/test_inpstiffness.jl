using TopOpt
using TopOpt.TopOptProblems.InputOutput.INP: InpStiffness, get_load_cells, getcloaddict, getfacesets, getpressuredict
using TopOpt.TopOptProblems: Metadata, getE, getν, getdensity, getgeomorder, nnodespercell
using Ferrite, Test

@testset "InpStiffness Tests" begin
    @testset "MBB.inp - 2D CPS4 elements" begin
        inp_file = joinpath(@__DIR__, "MBB.inp")
        
        # Test InpStiffness constructor from file path
        problem = InpStiffness(inp_file)
        
        # Check problem type
        @test problem isa InpStiffness
        @test problem isa StiffnessTopOptProblem{2, Float64}
        
        # Check material properties
        @test getE(problem) == 42000.0
        @test getν(problem) == 0.2
        @test getdensity(problem) == 0
        
        # Check geometry order (CPS4 is linear)
        @test getgeomorder(problem) == 1
        
        # Check nodes per cell (CPS4 has 4 nodes)
        @test nnodespercell(problem) == 4
        
        # Check grid properties
        grid = problem.ch.dh.grid
        @test getncells(grid) == 400  # 400 elements in MBB
        
        # Check constraint handler
        @test problem.ch isa ConstraintHandler
        
        # Check metadata
        @test problem.metadata isa Metadata
        
        # Test get_load_cells
        load_cells = get_load_cells(problem)
        @test load_cells isa Set{Int}
        # Load is applied at node 431, which should map to specific cells
        @test length(load_cells) > 0
        
        # Test getcloaddict
        cloads = getcloaddict(problem)
        @test cloads isa Dict
        @test haskey(cloads, 431)
        @test cloads[431] == [0.0, -3.0]
        
        # Test getfacesets
        facesets = getfacesets(problem)
        @test facesets isa Dict
        
        # Test getpressuredict
        dloads = getpressuredict(problem)
        @test dloads isa Dict
    end
    
    @testset "testcube.inp - 3D C3D10 elements" begin
        inp_file = joinpath(@__DIR__, "testcube.inp")
        
        # Test InpStiffness constructor from file path
        problem = InpStiffness(inp_file)
        
        # Check problem type
        @test problem isa InpStiffness
        @test problem isa StiffnessTopOptProblem{3, Float64}
        
        # Check material properties
        @test getE(problem) == 70_000
        @test getν(problem) == 0.3
        
        # Check geometry order (C3D10 is quadratic)
        @test getgeomorder(problem) == 2
        
        # Check nodes per cell (C3D10 has 10 nodes)
        @test nnodespercell(problem) == 10
        
        # Check grid properties
        grid = problem.ch.dh.grid
        @test getncells(grid) == 5
        
        # Test get_load_cells
        load_cells = get_load_cells(problem)
        @test load_cells isa Set{Int}
        
        # Test getcloaddict - concentrated load at node with coords (10, 10, 10)
        cloads = getcloaddict(problem)
        @test cloads isa Dict
        @test length(cloads) > 0
        
        # Get the force node and check its properties
        force_node = collect(keys(cloads))[1]
        @test problem.inp_content.node_coords[force_node] == (10, 10, 10)
        @test cloads[force_node] == [0, -1, 0]
        
        # Test getfacesets
        facesets = getfacesets(problem)
        @test haskey(facesets, "DLOAD_SET_1")
        @test facesets["DLOAD_SET_1"] == [(1, 3), (5, 2)]
        
        # Test getpressuredict
        dloads = getpressuredict(problem)
        @test haskey(dloads, "DLOAD_SET_1")
        @test dloads["DLOAD_SET_1"] == 1
    end
    
    @testset "get_load_cells functionality" begin
        inp_file = joinpath(@__DIR__, "MBB.inp")
        problem = InpStiffness(inp_file)
        
        # Get load cells
        load_cells = get_load_cells(problem)
        
        # Verify load cells are valid cell indices
        n_cells = getncells(problem.ch.dh.grid)
        for cell_id in load_cells
            @test 1 <= cell_id <= n_cells
        end
        
        # Verify that each load cell contains at least one loaded node
        cloads = getcloaddict(problem)
        for cell_id in load_cells
            # Get nodes in this cell
            cell = problem.ch.dh.grid.cells[cell_id]
            cell_nodes = collect(cell.nodes)
            
            # Check if any loaded node is in this cell
            has_load_node = any(node_id -> haskey(cloads, node_id), cell_nodes)
            @test has_load_node
        end
    end
    
    @testset "InpStiffness basic constructor" begin
        inp_file = joinpath(@__DIR__, "MBB.inp")
        
        # Test basic constructor
        problem = InpStiffness(inp_file)
        @test problem isa InpStiffness
        @test problem.ch isa ConstraintHandler
    end
    
    @testset "InpStiffness error handling" begin
        # Test with non-existent file
        @test_throws Exception InpStiffness("nonexistent_file.inp")
    end
    
    @testset "FEASolver construction" begin
        inp_file = joinpath(@__DIR__, "MBB.inp")
        problem = InpStiffness(inp_file)
        
        # Verify FEASolver can be constructed without error
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        @test !isnothing(solver)
    end
end

println("All InpStiffness tests completed!")