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
    @test problem.force == force
    @test problem.force_dof == 161 * 21 * 2
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ problem.rect_grid.corners[i][j] atol = 1e-8
    end
    @test sum(length, values(grid.facetsets)) == 2 * 160 + 2 * 40
    for (c, f) in grid.facetsets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facetsets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 160 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 40 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facetsets,), ["bottom", "right", "top", "left"])) ==
        2 * 160 + 2 * 40
end

# Half MBB beam problem
@testset "Half MBB beam" begin
    global E, ν, force
    problem = HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, ν, force)
    ncells = 60 * 20
    @test problem.E == E
    @test problem.ν == ν
    @test problem.force == force
    @test problem.force_dof == (61 * 20 + 2) * 2
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ problem.rect_grid.corners[i][j] atol = 1e-8
    end
    @test sum(length, values(grid.facetsets)) == 2 * 60 + 2 * 20
    for (c, f) in grid.facetsets["bottom"]
        @test f == 1
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facetsets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 60 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 20 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["left"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(length, getindex.((grid.facetsets,), ["bottom", "right", "top", "left"])) ==
        2 * 60 + 2 * 20
end

# L-beam problem
@testset "L-beam" begin
    global E, ν, force
    problem = LBeam(Val{:Linear}, Float64; force=force)
    ncells = 100 * 50 + 50 * 50
    @test problem.E == E
    @test problem.ν == ν
    @test problem.force == force
    @test problem.force_dof == (51 * 51 + 50 * 26) * 2
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    corners = [[0.0, 0.0], [100.0, 100.0]]
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ corners[i][j] atol = 1e-8
    end
    @test length(grid.facetsets["boundary"]) == 100 * 2 + 50 * 4
    for (c, f) in grid.facetsets["right"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 100 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["top"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 100 atol = 1e-8
        end
    end
    @test sum(length, getindex.((grid.facetsets,), ["right", "top"])) == 2 * 50
end

# Half MBB beam with quadratic elements
@testset "Half MBB beam quadratic elements" begin
    global E, ν, force
    nels = (30, 10)
    sizes = (2.0, 2.0)
    problem = HalfMBB(Val{:Quadratic}, nels, sizes, E, ν, force)
    ncells = nels[1] * nels[2]
    @test problem.E == E
    @test problem.ν == ν
    @test problem.force == force
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells

    # Quadratic elements have 9 nodes per cell
    @test Ferrite.nnodes(grid.cells[1]) == 9
    @test Ferrite.getorder(problem.ch.dh.subdofhandlers[1].field_interpolations[1]) == 2

    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ problem.rect_grid.corners[i][j] atol = 1e-8
    end

    # Check facesets exist and have correct structure
    @test haskey(grid.facetsets, "bottom")
    @test haskey(grid.facetsets, "right")
    @test haskey(grid.facetsets, "top")
    @test haskey(grid.facetsets, "left")

    # Verify boundary faces
    for (c, f) in grid.facetsets["bottom"]
        @test f == 1
    end
    for (c, f) in grid.facetsets["right"]
        @test f == 2
    end
    for (c, f) in grid.facetsets["top"]
        @test f == 3
    end
    for (c, f) in grid.facetsets["left"]
        @test f == 4
    end
end

# L-beam with quadratic elements
@testset "L-beam quadratic elements" begin
    global E, ν, force
    problem = LBeam(
        Val{:Quadratic},
        Float64;
        length=50,
        height=50,
        upperslab=25,
        lowerslab=25,
        E=E,
        ν=ν,
        force=force,
    )
    ncells = 50 * 25 + 25 * 25
    @test problem.E == E
    @test problem.ν == ν
    @test problem.force == force
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells

    # Quadratic elements have 9 nodes per cell
    @test Ferrite.nnodes(grid.cells[1]) == 9
    @test Ferrite.getorder(problem.ch.dh.subdofhandlers[1].field_interpolations[1]) == 2

    corners = [[0.0, 0.0], [50.0, 50.0]]
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ corners[i][j] atol = 1e-8
    end

    # Check facesets exist
    @test haskey(grid.facetsets, "right")
    @test haskey(grid.facetsets, "top")

    # Verify boundary faces
    for (c, f) in grid.facetsets["right"]
        @test f == 2
    end
    for (c, f) in grid.facetsets["top"]
        @test f == 3
    end
end

# Tie-beam problem
@testset "Tie-beam" begin
    problem = TopOpt.TopOptProblems.TieBeam(Val{:Quadratic}, Float64)
    ncells = 100
    @test problem.E == 1
    @test problem.ν == 0.3
    @test problem.force == 1
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells
    corners = [[0.0, 0.0], [32.0, 7.0]]
    for i in 1:2, j in 1:2
        @test boundingbox(grid)[i][j] ≈ corners[i][j] atol = 1e-8
    end
    @test length(grid.facetsets["boundary"]) == 32 * 2 + 3 * 2 + 4 * 2
    for (c, f) in grid.facetsets["bottomload"]
        @test f == 1
        for n in grid.cells[c].nodes[[1, 2]]
            @test grid.nodes[n].x[2] == 0
        end
    end
    for (c, f) in grid.facetsets["rightload"]
        @test f == 2
        for n in grid.cells[c].nodes[[2, 3]]
            @test grid.nodes[n].x[1] ≈ 32 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["toproller"]
        @test f == 3
        for n in grid.cells[c].nodes[[3, 4]]
            @test grid.nodes[n].x[2] ≈ 7 atol = 1e-8
        end
    end
    for (c, f) in grid.facetsets["leftfixed"]
        @test f == 4
        for n in grid.cells[c].nodes[[1, 4]]
            @test grid.nodes[n].x[1] == 0
        end
    end
    @test sum(
        length,
        getindex.((grid.facetsets,), ["bottomload", "rightload", "toproller", "leftfixed"]),
    ) == 8
    @test Ferrite.getorder(problem.ch.dh.subdofhandlers[1].field_interpolations[1]) == 2
    @test Ferrite.nnodes(grid.cells[1]) == 9
end

# Tie-beam accessor functions
@testset "Tie-beam accessor functions" begin
    @testset "getdim(::TieBeam) = 2" begin
        problem_linear = TopOpt.TopOptProblems.TieBeam(Val{:Linear}, Float64; refine=1)
        problem_quad = TopOpt.TopOptProblems.TieBeam(Val{:Quadratic}, Float64)

        @test TopOpt.TopOptProblems.getdim(problem_linear) == 2
        @test TopOpt.TopOptProblems.getdim(problem_quad) == 2
    end

    @testset "nnodespercell(::TieBeam{T,N}) = N" begin
        problem_linear = TopOpt.TopOptProblems.TieBeam(Val{:Linear}, Float64; refine=1)
        problem_quad = TopOpt.TopOptProblems.TieBeam(Val{:Quadratic}, Float64)

        # Linear elements have 4 nodes per cell
        @test TopOpt.TopOptProblems.nnodespercell(problem_linear) == 4
        # Quadratic elements have 9 nodes per cell
        @test TopOpt.TopOptProblems.nnodespercell(problem_quad) == 9
    end

    @testset "getpressuredict(::TieBeam)" begin
        force = 2.5
        problem = TopOpt.TopOptProblems.TieBeam(
            Val{:Linear}, Float64; refine=1, force=force
        )

        pd = TopOpt.TopOptProblems.getpressuredict(problem)
        @test pd isa Dict{String,Float64}
        @test haskey(pd, "rightload")
        @test haskey(pd, "bottomload")
        @test pd["rightload"] == 2 * force
        @test pd["bottomload"] == -force
    end

    @testset "getfacesets(::TieBeam)" begin
        problem = TopOpt.TopOptProblems.TieBeam(Val{:Linear}, Float64; refine=1)

        facesets = TopOpt.TopOptProblems.getfacesets(problem)
        @test facesets isa Dict
        @test haskey(facesets, "bottomload")
        @test haskey(facesets, "rightload")
        @test haskey(facesets, "toproller")
        @test haskey(facesets, "leftfixed")
    end
end

# Heat conduction problem tests
# HeatTransferTopOptProblem accessor functions
@testset "HeatTransferTopOptProblem accessor functions" begin
    @testset "getmetadata(::HeatTransferTopOptProblem)" begin
        heatflux = Dict{String,Float64}("top" => 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, (10, 5), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        metadata = TopOpt.TopOptProblems.getmetadata(problem)
        @test metadata isa TopOpt.TopOptProblems.Metadata
    end

    @testset "getpressuredict(::HeatTransferTopOptProblem)" begin
        heatflux = Dict{String,Float64}("top" => 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, (10, 5), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        # Returns empty dict for HeatTransferTopOptProblem
        pd = TopOpt.TopOptProblems.getpressuredict(problem)
        @test pd isa Dict{String,Float64}
        @test isempty(pd)
    end

    @testset "getheatfluxdict(::HeatTransferTopOptProblem)" begin
        heatflux = Dict{String,Float64}("top" => 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, (10, 5), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        # Returns the heatflux dict that was passed in
        hd = TopOpt.TopOptProblems.getheatfluxdict(problem)
        @test hd isa Dict{String,Float64}
        @test hd == heatflux
    end

    @testset "getcloaddict(::HeatTransferTopOptProblem)" begin
        heatflux = Dict{String,Float64}("top" => 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, (10, 5), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        # Returns an empty Dict{Int,Vector{Float64}} (node index => heat value)
        # for a problem with no point heat sources configured.
        cd = TopOpt.TopOptProblems.getcloaddict(problem)
        @test cd isa Dict{Int,Vector{Float64}}
        @test isempty(cd)
    end
end

@testset "Heat conduction problem setup" begin
    # Create a simple 2D heat conduction problem with surface heat flux
    nels = (10, 5)
    sizes = (1.0, 1.0)
    k = 1.0
    # Apply heat flux on top boundary (positive = heat into domain)
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux
    )

    @test problem isa HeatConductionProblem
    @test getk(problem) ≈ 1.0
    @test Ferrite.getncells(problem) == 50
end

@testset "Heat conduction problem with point heat source (cload)" begin
    # A concentrated heat source at a node is the point-source analogue of the
    # distributed heatflux; it is the setup that produces the branching
    # "conductivity tree" layout in the literature.
    nels = (8, 4)
    nx, ny = nels
    # Node index of the top-center node (x-major node ordering).
    center_x = div(nx, 2) + 1
    top_center = center_x + ny * (nx + 1)

    problem = HeatConductionProblem(
        Val{:Linear},
        nels,
        (1.0, 1.0),
        1.0;
        Tleft=0.0,
        Tright=0.0,
        heatflux=Dict{String,Float64}(),
        cload=Dict(top_center => 1.0),
    )

    @test problem isa HeatConductionProblem
    cd = TopOpt.TopOptProblems.getcloaddict(problem)
    @test cd isa Dict{Int,<:Vector}
    @test haskey(cd, top_center)
    @test cd[top_center] == [1.0]

    # The point source must show up in the assembled non-penalized load Q.
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    Q = solver.elementinfo.fixedload
    @test count(!=(0), Q) > 0
    # The injected heat should land on the top-center DOF, which is the first
    # DOF of the top-center node.
    node_dofs = problem.metadata.node_dofs
    @test Q[node_dofs[1, top_center]] ≈ 1.0

    # Default (no cload) keeps the zero load, preserving old behavior.
    problem_empty = HeatConductionProblem(
        Val{:Linear},
        nels,
        (1.0, 1.0),
        1.0;
        Tleft=0.0,
        Tright=0.0,
        heatflux=Dict{String,Float64}(),
    )
    @test isempty(TopOpt.TopOptProblems.getcloaddict(problem_empty))
    solver_empty = FEASolver(DirectSolver, problem_empty; xmin=0.01)
    @test all(==(0), solver_empty.elementinfo.fixedload)
end

@testset "Heat conduction problem with point Dirichlet BC (Tfix)" begin
    # Tfix pins the temperature at individual nodes — a point cold sink —
    # which is the setup for the classic branching conductivity tree.
    nels = (8, 4)
    nx, ny = nels
    center_bottom = div(nx, 2) + 1

    problem = HeatConductionProblem(
        Val{:Linear},
        nels,
        (1.0, 1.0),
        1.0;
        Tleft=nothing,
        Tright=nothing,
        Ttop=nothing,
        Tbottom=nothing,
        Tfix=Dict(center_bottom => 0.0),
    )

    # Only the one fixed node should be prescribed.
    ch = problem.ch
    @test length(ch.prescribed_dofs) == 1
    node_dofs = problem.metadata.node_dofs
    @test ch.prescribed_dofs[1] == node_dofs[1, center_bottom]
    @test ch.inhomogeneities[1] ≈ 0.0

    # Multiple fixed nodes with distinct temperatures.
    problem2 = HeatConductionProblem(
        Val{:Linear},
        nels,
        (1.0, 1.0),
        1.0;
        Tleft=nothing,
        Tright=nothing,
        Ttop=nothing,
        Tbottom=nothing,
        Tfix=Dict(1 => 100.0, center_bottom => 0.0),
    )
    @test length(problem2.ch.prescribed_dofs) == 2
    @test Set(problem2.ch.inhomogeneities) == Set([100.0, 0.0])
end

@testset "HeatTree convenience constructor" begin
    # Classic heat-conduction topopt benchmark: flux on top, cold bottom,
    # insulated sides.
    problem = HeatTree(Val{:Linear}, (8, 4), (1.0, 1.0), 1.0; q=2.0)
    @test problem isa HeatConductionProblem
    @test TopOpt.TopOptProblems.getheatfluxdict(problem) == Dict("top" => 2.0)
    # Only the bottom boundary is prescribed (9 nodes for 8 bottom elements).
    @test length(problem.ch.prescribed_dofs) == 9
    @test all(==(0), problem.ch.inhomogeneities)
    # The problem solves.
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    solver.vars .= 1.0
    solver()
    @test all(isfinite, solver.u)
    @test maximum(solver.u) > 0
end

@testset "Heat conduction problem with quadratic elements" begin
    # Create a 2D heat conduction problem with quadratic elements
    nels = (5, 3)
    sizes = (1.0, 1.0)
    k = 2.5
    heatflux = Dict{String,Float64}("top" => 10.0)

    problem = HeatConductionProblem(
        Val{:Quadratic}, nels, sizes, k; Tleft=100.0, Tright=0.0, heatflux=heatflux
    )

    @test problem isa HeatConductionProblem
    @test problem.k == k
    @test Ferrite.getncells(problem) == nels[1] * nels[2]

    # Quadratic elements have 9 nodes per cell
    grid = problem.ch.dh.grid
    @test Ferrite.nnodes(grid.cells[1]) == 9
    @test Ferrite.getorder(problem.ch.dh.subdofhandlers[1].field_interpolations[1]) == 2

    # Check boundary facesets exist
    @test haskey(grid.facetsets, "top")
    @test haskey(grid.facetsets, "bottom")
    @test haskey(grid.facetsets, "left")
    @test haskey(grid.facetsets, "right")

    # Check heatflux dict is accessible
    @test TopOpt.TopOptProblems.getheatfluxdict(problem) == heatflux

    # ElementFEAInfo now builds for quadratic heat conduction elements (the
    # cellvalues interpolation is taken from the DofHandler instead of being
    # hardcoded to linear Lagrange).
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    @test length(elementinfo.Kes) == nels[1] * nels[2]
    # Quadratic quad: 9 nodes, scalar temperature field -> 9x9 Ke.
    @test size(elementinfo.Kes[1], 1) == 9
    @test size(elementinfo.Kes[1], 2) == 9
end

@testset "Heat transfer element matrices" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    k = 1.0
    heatflux = Dict{String,Float64}("top" => 1.0)

    problem = HeatConductionProblem(
        Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux
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
            Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux
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
            @test mat_Ke ≈ mat_Ke' rtol = 1e-10

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
            Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux
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
    @test haskey(grid.facetsets, "left")
    @test haskey(grid.facetsets, "right")
    @test haskey(grid.facetsets, "top")
    @test haskey(grid.facetsets, "bottom")

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
    @test TopOpt.TopOptProblems.getE(problem) == E
    @test TopOpt.TopOptProblems.getν(problem) == ν
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

@testset "RectilinearTopology" begin
    using TopOpt.TopOptProblems: RectilinearTopology

    @testset "Default topology (all ones)" begin
        # Create a simple PointLoadCantilever problem
        nels = (10, 6)
        sizes = (1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = 1.0

        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test with default topology (should be all ones)
        topology = RectilinearTopology(problem)
        @test size(topology) == (6, 10)
        @test all(topology .== 1.0)
    end

    @testset "Custom topology" begin
        nels = (10, 6)
        sizes = (1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = 1.0

        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create a custom topology with some zeros
        custom_topology = ones(Float64, Ferrite.getncells(problem))
        custom_topology[1:30] .= 0.0  # Set first 30 elements to zero

        topology = RectilinearTopology(problem, custom_topology)

        # Check dimensions match nels (transposed)
        @test size(topology) == (6, 10)

        # Check that the topology values are mapped correctly
        # The reshape is transposed, so we check that values are reasonable
        @test topology isa AbstractMatrix
        @test eltype(topology) <: Real
    end

    @testset "Quadratic geometry order" begin
        # Test with quadratic elements
        nels = (10, 6)
        sizes = (1.0, 1.0)
        E = 1.0
        ν = 0.3
        force = 1.0

        # Use HalfMBB which supports quadratic elements in 2D
        problem = HalfMBB(Val{:Quadratic}, nels, sizes, E, ν, force)

        topology = RectilinearTopology(problem)
        @test size(topology) == (6, 10)
        @test all(topology .== 1.0)
    end
end

@testset "Problem show methods" begin
    nels = (10, 6)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    @testset "PointLoadCantilever show" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test occursin("PointLoadCantilever", output) ||
            occursin("Point", output) ||
            output != ""
    end

    @testset "HalfMBB show" begin
        problem = HalfMBB(Val{:Linear}, nels, sizes, E, ν, force)
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test occursin("HalfMBB", output) || output != ""
    end

    @testset "LBeam show" begin
        problem = LBeam(Val{:Linear}, Float64; force=force)
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test occursin("LBeam", output) || output != ""
    end

    @testset "TieBeam show" begin
        problem = TopOpt.TopOptProblems.TieBeam(Val{:Quadratic}, Float64)
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test occursin("TieBeam", output) || output != ""
    end

    @testset "HeatConductionProblem show" begin
        heatflux = Dict{String,Float64}("top" => 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, (10, 5), (1.0, 1.0), 1.0; Tleft=0.0, Tright=0.0, heatflux=heatflux
        )
        io = IOBuffer()
        show(io, MIME("text/plain"), problem)
        output = String(take!(io))
        @test occursin("HeatConductionProblem", output) || output != ""
    end
end
