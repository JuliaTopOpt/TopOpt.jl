using TopOpt
using TopOpt.TopOptProblems
using TopOpt.TrussTopOptProblems
using Test
using LinearAlgebra
using SparseArrays
using Ferrite

# Import types needed for TrussElementKσ evaluation
using TopOpt.Functions: DisplacementResult
using TopOpt: PseudoDensities

# Test assembly force functions
@testset "Assembly force functions" begin
    # Create a simple cantilever problem
    nels = (4, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    # Test assemble_f! with existing vector (main API)
    @testset "assemble_f! inplace" begin
        penalty = PowerPenalty(3.0)
        vars = ones(Float64, ncells)

        # Create output vector
        f_out = zeros(Float64, ndofs(problem.ch.dh))

        # Call inplace version
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)

        @test length(f_out) == ndofs(problem.ch.dh)
        @test any(f_out .!= 0)  # Should have been filled with values
    end

    # Test assemble_f! with distributed loads
    @testset "assemble_f! with distributed loads" begin
        # This test covers the second assemble_f! method that takes dloads
        metadata = problem.metadata
        dof_cells = metadata.dof_cells

        # Create simple dloads
        dloads = [rand(Float64, 8) for _ in 1:ncells]

        f = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f, problem, dloads)

        @test length(f) == ndofs(problem.ch.dh)
    end

    # Test update_f! with dof_cells (2-argument version)
    @testset "update_f! with dof_cells" begin
        metadata = problem.metadata
        dof_cells = metadata.dof_cells

        # Create a simple dloads structure
        dloads = [zeros(Float64, 8) for _ in 1:ncells]  # 8 DOFs per cell for 2D quad

        # Initialize f vector
        f = zeros(Float64, ndofs(problem.ch.dh))

        # Call update_f! (2-argument version)
        TopOpt.TopOptProblems.update_f!(f, dof_cells, dloads)

        # Since dloads is all zeros, f should remain zero
        @test all(f .== 0)
    end
end

# Test buckling analysis functions
@testset "Buckling analysis" begin
    # Create a simple cantilever problem for buckling
    nels = (4, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

    # Create global FEA info
    ginfo = GlobalFEAInfo(problem)

    # Assemble the stiffness matrix
    vars = ones(Float64, getncells(problem.ch.dh.grid))
    assemble!(ginfo, problem, elementinfo, vars)

    @testset "get_Kσs function" begin
        # Test that get_Kσs returns the geometric stiffness matrices
        # Need actual displacement values from solving the system
        u = ginfo.K \ ginfo.f
        Kσs = TopOpt.TopOptProblems.get_Kσs(problem, u, elementinfo.cellvalues)

        ncells = getncells(problem.ch.dh.grid)
        @test length(Kσs) == ncells

        # Each Kσ should be a matrix
        for Kσ in Kσs
            @test Kσ isa Matrix
            @test size(Kσ, 1) == size(Kσ, 2)  # Square matrices
        end
    end

    @testset "buckling function" begin
        # Test the buckling function
        K, Kσ = buckling(problem, ginfo, elementinfo)

        # Both should be arrays/matrices (buckling returns matrix-like structures)
        @test K isa AbstractArray
        @test Kσ isa AbstractArray
        @test ndims(K) == 2
        @test ndims(Kσ) == 2

        # Both should be the same size
        @test size(K) == size(Kσ)

        # K should be positive definite (for a valid structural problem)
        # We'll check this by verifying it's symmetric and has positive diagonal
        @test all(diag(K) .> 0)
    end
end

# Test TrussElementKσ for truss problems
@testset "Truss buckling - TrussElementKσ" begin
    using TopOpt.TrussTopOptProblems
    using TopOpt.TrussTopOptProblems: getE, getA, compute_local_axes

    # Load a simple truss problem
    ins_dir = joinpath(@__DIR__, "..", "truss_topopt_problems", "instances", "fea_examples")
    file_name = "buckling_2d_nodal_instab.json"
    problem_file = joinpath(ins_dir, file_name)

    if isfile(problem_file)
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
            problem_file
        )
        ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
        loads = load_cases["0"]

        problem = TrussProblem(
            Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
        )

        # Create solver
        solver = FEASolver(DirectSolver, problem)
        solver()

        # Create TrussElementKσ
        @testset "TrussElementKσ construction" begin
            eksig = TrussElementKσ(problem, solver)

            @test eksig isa TopOpt.Functions.TrussElementKσ
            @test hasfield(typeof(eksig), :problem)
            @test hasfield(typeof(eksig), :Kσes)
            @test hasfield(typeof(eksig), :EALγ_s)
        end

        @testset "TrussElementKσ evaluation" begin
            eksig = TrussElementKσ(problem, solver)

            # Test with displacement result - wrap in proper types
            u_result = TopOpt.Functions.DisplacementResult(solver.u)
            x_densities = TopOpt.PseudoDensities(ones(ncells))

            # Call the evaluator
            Kσ_result = eksig(u_result, x_densities)

            @test Kσ_result isa AbstractVector  # Should return element Kσ matrices
            @test length(Kσ_result) == ncells
        end

        @testset "TrussElementKσ show method" begin
            eksig = TrussElementKσ(problem, solver)
            io = IOBuffer()
            show(io, MIME("text/plain"), eksig)
            output = String(take!(io))
            @test occursin("stress stiffness", output) || occursin("Kσ", output)
        end
    else
        @warn "Truss problem file not found, skipping TrussElementKσ tests"
    end
end

# Test assembly with different penalty schemes
@testset "Assembly with penalty variations" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    @testset "Power penalty" begin
        vars = fill(0.5, ncells)
        penalty = PowerPenalty(3.0)
        f_out = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)
        @test length(f_out) == ndofs(problem.ch.dh)
    end

    @testset "Rational penalty" begin
        vars = fill(0.5, ncells)
        penalty = RationalPenalty(3.0)
        f_out = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)
        @test length(f_out) == ndofs(problem.ch.dh)
    end
end

# Test error handling and edge cases
@testset "Assembly edge cases" begin
    nels = (2, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    @testset "Very small densities" begin
        vars = fill(0.001, ncells)
        penalty = PowerPenalty(3.0)
        f_out = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)
        @test length(f_out) == ndofs(problem.ch.dh)
        @test all(isfinite, f_out)
    end
end
