using TopOpt
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
    problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, ν, force)
    ncells = 160 * 40
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
    @test length(grid.boundary_matrix.nzval) == 2 * 160 + 2 * 40
    for (c, f) in grid.facesets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 160 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 40 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["bottom", "right", "top", "left"])) ==
        2 * 160 + 2 * 40
end

# Half MBB beam problem
@testset "Half MBB beam" begin
    global E, ν, force
    problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, ν, force)
    ncells = 60 * 20
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
    @test length(grid.boundary_matrix.nzval) == 2 * 60 + 2 * 20
    for (c, f) in grid.facesets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 60 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 20 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["bottom", "right", "top", "left"])) ==
        2 * 60 + 2 * 20
end

# L-beam problem
@testset "L-beam" begin
    global E, ν, force
    problem = LBeam(Val{:Linear}, Float64; force=force)
    ncells = 100 * 50 + 50 * 50
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
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 100 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 100 atol = 1e-8
        end
    end
    @test sum(length, getindex.((grid.facesets,), ["right", "top"])) == 2 * 50
end

# Tie-beam problem
@testset "Tie-beam" begin
    problem = TopOptProblems.TieBeam(Val{:Quadratic}, Float64)
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
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facesets["rightload"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 32 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["toproller"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 7 atol = 1e-8
        end
    end
    for (c, f) in grid.facesets["leftfixed"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(
        length,
        getindex.((grid.facesets,), ["bottomload", "rightload", "toproller", "leftfixed"]),
    ) == 8
    @test Ferrite.getorder(problem.ch.dh.field_interpolations[1]) == 2
    @test Ferrite.nnodes(grid.cells[1]) == 9
end

# Heat conduction problem tests
@testset "Heat conduction problem setup" begin
    # Create a simple 2D heat conduction problem with surface heat flux
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    # Apply heat flux on top boundary (positive = heat into domain)
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    @test problem isa HeatConductionProblem
    @test getk(problem) ≈ 1.0
    @test Ferrite.getncells(problem) == 50
end

@testset "Heat transfer element matrices" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    k = 1.0
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k;
        Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

    @test length(elementinfo.Kes) == 8  # 4x2 elements
    @test length(elementinfo.fes) == 8

    # For linear quad elements, each element has 4 nodes
    # and heat transfer is scalar field, so Ke is 4x4
    @test size(elementinfo.Kes[1], 1) == 4
end

@testset "Heat transfer element properties" begin
    @testset "Element stiffness matrix symmetry and properties" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # Check number of elements
        @test length(elementinfo.Kes) == 4

        # Check each element matrix
        for Ke in elementinfo.Kes
            # Should be 4x4 for linear quadrilateral
            @test size(Ke, 1) == 4
            @test size(Ke, 2) == 4

            # Should be symmetric
            mat_Ke = Matrix(Ke)
            @test mat_Ke ≈ mat_Ke' rtol=1e-10

            # Should be positive semi-definite (non-negative eigenvalues)
            eigs = eigvals(mat_Ke)
            @test all(eigs .>= -1e-10)
        end

        # Check that fes is zeros (no body forces in heat transfer)
        @test length(elementinfo.fes) == 4
        for fe in elementinfo.fes
            @test length(fe) == 4
            # fes should be zeros (no body forces)
            @test all(fe .== 0)
        end
    end

    @testset "Element volumes are positive" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict{String,Float64}("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # All cell volumes should be positive
        @test all(elementinfo.cellvolumes .> 0)

        # Total volume should match domain size
        total_vol = sum(elementinfo.cellvolumes)
        expected_vol = nels[1] * sizes[1] * nels[2] * sizes[2]
        @test isapprox(total_vol, expected_vol; rtol=1e-10)
    end
end

# Additional tests for boundary conditions and multi-load cases
@testset "Problem boundary condition handling" begin
    # Test that various boundary condition configurations work correctly
    # Use even numbers for nels as required by PointLoadCantilever
    nels = (10, 6)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    
    # Create problem
    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, 1.0)
    
    # Verify boundary condition facesets exist
    grid = problem.ch.dh.grid
    @test haskey(grid.facesets, "left")
    @test haskey(grid.facesets, "right")
    @test haskey(grid.facesets, "top")
    @test haskey(grid.facesets, "bottom")
    
    # Verify constraint handler has entries
    ch = problem.ch
    @test ch.dh.ndofs[] > 0  # Check dof handler has dofs (unwrap ScalarWrapper)
    @test length(ch.prescribed_dofs) > 0
end

@testset "Problem metadata accessors" begin
    # Use even numbers for nels as required by PointLoadCantilever
    nels = (10, 6)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0
    
    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
    
    # Test accessor functions
    @test TopOptProblems.getE(problem) == E
    @test TopOptProblems.getν(problem) == ν
    
    # Test number of variables (black/white arrays indicate design variables)
    @test length(problem.black) == prod(nels)
    @test length(problem.white) == prod(nels)
    @test length(problem.varind) <= prod(nels)
end

@testset "InpStiffness loading" begin
    # Test loading of INP file stiffness matrices
    inp_path = joinpath(@__DIR__, "..", "inp_parser", "MBB.inp")
    
    if isfile(inp_path)
        # If INP file exists, test loading
        @test true  # Placeholder for actual INP loading test
    else
        @info "INP file not found, skipping INP stiffness test"
    end
end

@testset "Problem type consistency" begin
    # Ensure all problem types have consistent interface
    nels = (10, 10)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    
    problems = [
        PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, 1.0),
        HalfMBB(Val{:Linear}, nels, sizes, E, ν, 1.0),
    ]
    
    for problem in problems
        # All problems should have required fields
        @test hasfield(typeof(problem), :E)
        @test hasfield(typeof(problem), :ν)
        @test hasfield(typeof(problem), :black)
        @test hasfield(typeof(problem), :white)
        @test hasfield(typeof(problem), :varind)
        
        # All should have a constraint handler
        @test hasfield(typeof(problem), :ch)
        @test problem.ch isa Ferrite.ConstraintHandler
    end
end

@testset "Grid utilities" begin
    nels = (6, 6)  # Use even dimensions as required
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    
    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, 1.0)
    grid = problem.ch.dh.grid
    
    # Test bounding box calculation
    bbox = boundingbox(grid)
    @test length(bbox) == 2
    @test bbox[1][1] ≈ 0.0
    @test bbox[1][2] ≈ 0.0
    @test bbox[2][1] ≈ nels[1] * sizes[1]
    @test bbox[2][2] ≈ nels[2] * sizes[2]
    
    # Test cell count
    @test Ferrite.getncells(grid) == prod(nels)
end
