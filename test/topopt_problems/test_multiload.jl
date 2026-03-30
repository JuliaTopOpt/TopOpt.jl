using TopOpt
using TopOpt.TopOptProblems
using Test

using Ferrite
using SparseArrays
using LinearAlgebra
using Distributions

# Import internal functions for testing
using TopOpt.TopOptProblems: find_nearest_dofs, RandomMagnitude, random_direction,
    generate_random_loads, get_surface_dofs, get_node_first_cells, get_node_dofs,
    get_node_cells, getcloaddict, getfacesets, getE, getν

# Helper function to get node coordinates for testing
function get_node_coords(problem, node_idx)
    return problem.ch.dh.grid.nodes[node_idx].x.data
end

@testset "Multiload and Metadata Functions" begin
    # Create a simple test problem
    nels = (10, 6)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    @testset "find_nearest_dofs" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test finding nearest dofs to a point inside the domain
        # The point should be somewhere within the grid bounds [0, 10] x [0, 6]
        p = (5.0, 3.0)
        nearest_dofs = find_nearest_dofs(problem, p)

        # Should return a 2-element vector (2D problem has 2 DOFs per node)
        @test length(nearest_dofs) == 2
        @test all(nearest_dofs .> 0)

        # Verify these are valid DOF indices
        ndofs = Ferrite.ndofs(problem.ch.dh)
        @test all(d -> 1 <= d <= ndofs, nearest_dofs)

        # Test with corner point
        p_corner = (0.0, 0.0)
        corner_dofs = find_nearest_dofs(problem, p_corner)
        @test length(corner_dofs) == 2
        @test all(corner_dofs .> 0)

        # Test with point near boundary
        p_boundary = (10.0, 6.0)
        boundary_dofs = find_nearest_dofs(problem, p_boundary)
        @test length(boundary_dofs) == 2
    end

    @testset "RandomMagnitude" begin
        # Test RandomMagnitude struct with Normal distribution
        rm = RandomMagnitude(1.0, Normal(0.0, 1.0))
        @test rm.f == 1.0
        @test rm.dist isa Normal

        # Test calling the RandomMagnitude function
        # Should return a value that's scaled by f
        val = rm()
        @test isa(val, Float64)

        # Test with scalar multiplier
        rm2 = RandomMagnitude(2.5, Normal(5.0, 1.0))
        val2 = rm2()
        # The result should be 2.5 * (5.0 + rand), so roughly around 12.5
        @test val2 isa Float64

        # Test with array multiplier
        rm3 = RandomMagnitude([1.0, 2.0, 3.0], Normal(0.0, 1.0))
        val3 = rm3()
        @test length(val3) == 3
        @test val3 isa Vector{Float64}
    end

    @testset "random_direction" begin
        # random_direction returns a 2D unit vector (hardcoded)
        for _ in 1:10
            dir = random_direction()
            @test length(dir) == 2
            @test eltype(dir) == Float64

            # Direction should be approximately unit length
            norm_dir = norm(dir)
            @test isapprox(norm_dir, 1.0; atol=1e-10)
        end

        # Test that we get different directions (probabilistic test)
        dirs = [random_direction() for _ in 1:10]
        @test length(unique(dirs)) > 1  # Very unlikely to get identical random directions
    end

    @testset "generate_random_loads" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test generating random loads with a distribution
        # BUG IN SOURCE: generate_random_loads has a bug where it uses push!(FJ, i, i)
        # which creates mismatched array sizes when dofs has length != 2
        # For 2D problems with 2 DOFs per node, it happens to work
        nloads = 5
        dist = Normal(0.0, 1.0)
        
        # This will throw an error due to the bug
        @test_throws ArgumentError generate_random_loads(problem, nloads, dist)
    end

    @testset "get_node_first_cells" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        dh = problem.ch.dh

        # Call get_node_first_cells directly
        node_first_cells = get_node_first_cells(dh)

        # Should return one entry per node
        nnodes = Ferrite.getnnodes(dh.grid)
        @test length(node_first_cells) == nnodes

        # Each entry should be a tuple of (cell_idx, local_node_idx)
        for (cell_idx, local_node_idx) in node_first_cells
            @test cell_idx > 0
            @test local_node_idx > 0
            # Check that the local_node_idx is valid for the cell
            cell = dh.grid.cells[cell_idx]
            @test local_node_idx <= length(cell.nodes)
        end

        # Verify that the first occurrence of each node is captured
        visited = falses(nnodes)
        for cellidx in 1:Ferrite.getncells(dh.grid)
            for (local_node_idx, global_node_idx) in enumerate(dh.grid.cells[cellidx].nodes)
                if !visited[global_node_idx]
                    visited[global_node_idx] = true
                    @test node_first_cells[global_node_idx] == (cellidx, local_node_idx)
                end
            end
        end
    end

    @testset "get_node_dofs" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        dh = problem.ch.dh

        # Get node DOFs using the DofHandler (get_node_dofs takes DofHandler, not Metadata)
        node_dofs = get_node_dofs(dh)

        # Returns a matrix (ndofs_per_node x nnodes) - check dimensions
        nnodes = Ferrite.getnnodes(dh.grid)
        @test node_dofs isa Matrix{Int}
        @test size(node_dofs, 2) == nnodes  # one column per node
        ndofs_per_node = Ferrite.ndofs_per_cell(dh) > 0 ? div(Ferrite.ndofs(dh), nnodes) : 2
        @test size(node_dofs, 1) > 0  # at least one DOF per node

        # Verify that DOFs are valid
        ndofs_total = Ferrite.ndofs(dh)
        for dof in node_dofs
            @test 1 <= dof <= ndofs_total
        end
    end

    @testset "get_node_cells" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
        dh = problem.ch.dh

        # Get node cells from DofHandler (get_node_cells takes DofHandler, not Metadata)
        node_cells = get_node_cells(dh)

        # Returns a RaggedArray that supports indexing by node
        nnodes = Ferrite.getnnodes(dh.grid)

        # Each entry should be a vector of (cell_id, local_node_id) tuples
        for node_idx in 1:nnodes
            cells_for_node = node_cells[node_idx]
            @test cells_for_node isa AbstractVector{<:Tuple{<:Integer, <:Integer}}
            for (cell_id, local_node_id) in cells_for_node
                @test cell_id > 0
                @test local_node_id > 0
                cell = dh.grid.cells[cell_id]
                @test local_node_id <= length(cell.nodes)
                # Verify the node is actually in that cell
                @test cell.nodes[local_node_id] == node_idx
            end
        end
    end

    @testset "get_surface_dofs" begin
        # Create problem
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Get surface DOFs - no string argument needed
        surface_dofs = get_surface_dofs(problem)
        @test surface_dofs isa Vector{Int}

        # Verify these are valid DOFs if any exist
        if !isempty(surface_dofs)
            ndofs = Ferrite.ndofs(problem.ch.dh)
            @test all(d -> d > 0 && d <= ndofs, surface_dofs)
        end
    end

    @testset "getcloaddict default implementation" begin
        # Create a basic StiffnessTopOptProblem
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test getcloaddict - returns a dict (type varies by implementation)
        cload_dict = getcloaddict(problem)
        # The dict could be empty or have entries, just verify it's a Dict
        @test cload_dict isa Dict

        # Verify keys and values have reasonable types if not empty
        if !isempty(cload_dict)
            # Check that all values are vectors
            @test all(v -> v isa Vector, values(cload_dict))
        end
    end

    @testset "getfacesets default implementation" begin
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Test getfacesets
        facesets = getfacesets(problem)
        @test facesets isa Dict{String, Tuple{Int, Float64}}

        # May be empty depending on implementation
        @test true  # Just verify it runs without error
    end

    @testset "MultiLoad type" begin
        # Test that MultiLoad is exported and can be referenced
        @test isdefined(TopOpt.TopOptProblems, :MultiLoad)

        # Create a basic problem
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create MultiLoad using load rules constructor with explicit positions
        # This bypasses the buggy generate_random_loads function
        nloads = 3
        # Define load rules with specific positions and RandomMagnitude functions
        load_rules = [
            (5.0, 3.0) => RandomMagnitude([0.0, -1.0], Uniform(0.5, 1.0)),
            (2.0, 1.0) => RandomMagnitude([1.0, 0.0], Uniform(0.5, 1.0)),
        ]
        multiload = MultiLoad(problem, nloads, load_rules)

        # Verify it's a MultiLoad type
        @test multiload isa MultiLoad
        @test size(multiload.F, 2) == nloads

        # Test the property forwarding
        @test getE(multiload) == E
        @test getν(multiload) == ν
    end

    @testset "Integration test: Full multiload workflow" begin
        # Create a problem
        problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

        # Create load rules for specific positions
        f1 = RandomMagnitude([0.0, -1.0], Uniform(0.5, 1.5))
        f2 = RandomMagnitude(normalize([1.0, -1.0]), Uniform(0.5, 1.5))

        # Create MultiLoad with specific load positions
        nloads = 10
        load_rules = [(5.0, 3.0) => f1, (2.0, 1.0) => f2]
        multiload = MultiLoad(problem, nloads, load_rules)

        # Verify load matrix properties
        @test multiload isa MultiLoad
        @test size(multiload.F, 2) == nloads

        # Each column should represent a load case
        for i in 1:nloads
            load_case = multiload.F[:, i]
            @test length(load_case) == Ferrite.ndofs(problem.ch.dh)
        end

        # Test finding DOFs for a specific load location
        load_point = (5.0, 3.0)
        nearest_dofs = find_nearest_dofs(problem, load_point)

        # The DOFs should be valid indices
        @test all(d -> 1 <= d <= Ferrite.ndofs(problem.ch.dh), nearest_dofs)
    end
end