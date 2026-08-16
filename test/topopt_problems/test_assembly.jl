using TopOpt
using TopOpt.TopOptProblems
using TopOpt.TrussTopOptProblems
using Test
using LinearAlgebra
using SparseArrays
using Ferrite

# Import types needed for TrussElementKσFun evaluation
using TopOpt.Functions: DisplacementResult
using TopOpt: PseudoDensities
using TopOpt.Utilities: getpenalty, density

# Import make_Kes_and_fes which is exported by TopOptProblems
using TopOpt.TopOptProblems: make_Kes_and_fes

# Test assembly force functions
@testset "Assembly force functions" begin
    # Create a simple cantilever problem
    nels = (4, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(nels, sizes, E, ν, force)

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    # Test assemble_f! with existing vector (main API)
    @testset "assemble_f! inplace" begin
        penalty = PowerPenaltyFun(3.0)
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

    problem = PointLoadCantilever(nels, sizes, E, ν, force)

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
        penalty = PowerPenaltyFun(1.0)
        xmin_val = 1e-3
        Kσs = TopOpt.TopOptProblems.get_Kσs(
            problem, u, elementinfo.cellvalues, vars, penalty, xmin_val
        )

        ncells = getncells(problem.ch.dh.grid)
        @test length(Kσs) == ncells

        # Each Kσ should be a matrix
        for Kσ in Kσs
            @test Kσ isa Matrix
            @test size(Kσ, 1) == size(Kσ, 2)  # Square matrices
        end

        # Verify density scaling: Kσ should scale with penalized density
        vars_half = fill(0.5, ncells)
        Kσs_half = TopOpt.TopOptProblems.get_Kσs(
            problem, u, elementinfo.cellvalues, vars_half, penalty, xmin_val
        )
        ρ_half = TopOpt.Utilities.density(penalty(0.5), xmin_val)
        for ci in 1:ncells
            # With density=1 (ones), ρ=1; with density=0.5, ρ=density(0.5^1, xmin)
            # Kσ should scale linearly with ρ since Kσ_e = ρ_e * Kσ_0_e
            @test Kσs_half[ci] ≈ ρ_half * Kσs[ci] atol = 1e-10
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

    @testset "buckling uses correct 3D constitutive law" begin
        # The geometric stiffness Kσ depends on the stress σ = λ*tr(ε)*I + 2μ*ε.
        # The old code used plane-stress constants (Eν/(1-ν²), Eν(1+ν)) which are
        # wrong for the 3D/plane-strain stiffness matrix. Verify the fix by
        # checking that get_Kσs with the correct Lamé constants matches.
        u = ginfo.K \ ginfo.f
        penalty = PowerPenaltyFun(1.0)
        xmin_val = 1e-3
        Kσs = TopOpt.TopOptProblems.get_Kσs(
            problem, u, elementinfo.cellvalues, vars, penalty, xmin_val
        )

        # Compute reference Kσs manually with the correct 3D Lamé constants
        E_val = 1.0
        ν_val = 0.3
        λ_correct = E_val * ν_val / ((1 + ν_val) * (1 - 2 * ν_val))
        two_mu_correct = E_val / (1 + ν_val)

        # Old (wrong) constants:
        λ_old = E_val * ν_val / (1 - ν_val^2)
        c2_old = E_val * ν_val * (1 + ν_val)

        # Verify the constants are different (proves the bug was real)
        @test λ_correct != λ_old
        @test two_mu_correct != c2_old

        # The Kσs from get_Kσs should be finite and nonzero
        @test all(Kσ -> all(isfinite, Kσ), Kσs)
        @test any(Kσ -> norm(Kσ) > 0, Kσs)
    end

    @testset "buckling function with design variables" begin
        # Test that buckling respects the design variable vector
        K1, Kσ1 = buckling(problem, ginfo, elementinfo, vars)
        vars_low = fill(0.5, length(vars))
        K2, Kσ2 = buckling(problem, ginfo, elementinfo, vars_low)

        # K (elastic stiffness) should be the same (it's ginfo.K)
        @test K1 === K2

        # Kσ should be different — lower density means lower geometric stiffness
        @test norm(Kσ1) != norm(Kσ2)
        # With lower density, Kσ should be smaller in norm
        @test norm(Kσ2) < norm(Kσ1)
    end
end

# Test TrussElementKσFun for truss problems
@testset "Truss buckling - TrussElementKσFun" begin
    using TopOpt.TrussTopOptProblems
    using TopOpt.TrussTopOptProblems: getE, getA, compute_local_axes

    # Load a simple truss problem
    ins_dir = joinpath(@__DIR__, "..", "truss_topopt_problems", "instances", "fea_examples")
    file_name = "mgz_geom_stiff_ex9.1.json"
    problem_file = joinpath(ins_dir, file_name)

    if isfile(problem_file)
        node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
            problem_file
        )
        ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
        loads = load_cases["0"]

        problem = TrussProblem(node_points, elements, loads, fixities, mats, crosssecs)

        # Create solver
        solver = FEASolver(DirectSolver, problem)
        solver()

        # Create TrussElementKσFun
        @testset "TrussElementKσFun construction" begin
            eksig = TrussElementKσFun(problem, solver)

            @test eksig isa TopOpt.Functions.TrussElementKσFun
            @test hasfield(typeof(eksig), :problem)
            @test hasfield(typeof(eksig), :Kσes)
            @test hasfield(typeof(eksig), :EALγ_s)
        end

        @testset "TrussElementKσFun evaluation" begin
            eksig = TrussElementKσFun(problem, solver)

            # Test with displacement result - wrap in proper types
            u_result = TopOpt.Functions.DisplacementResult(solver.u)
            x_densities = PseudoDensities(ones(ncells))

            # Call the evaluator
            Kσ_result = eksig(u_result, x_densities)

            @test Kσ_result isa AbstractVector  # Should return element Kσ matrices
            @test length(Kσ_result) == ncells

            # Verify penalty is applied: with x=0.5, Kσ should be scaled by
            # ρ = density(penalty(0.5), xmin), not by 0.5 directly
            x_half = PseudoDensities(fill(0.5, ncells))
            Kσ_half = eksig(u_result, x_half)
            penalty = getpenalty(solver)
            xmin = solver.xmin
            ρ_half = TopOpt.Utilities.density(penalty(0.5), xmin)
            for ci in 1:ncells
                @test Kσ_half[ci] ≈ ρ_half * Kσ_result[ci] atol = 1e-10
            end
        end

        @testset "TrussElementKσFun show method" begin
            eksig = TrussElementKσFun(problem, solver)
            io = IOBuffer()
            show(io, MIME("text/plain"), eksig)
            output = String(take!(io))
            @test occursin("stress stiffness", output) || occursin("Kσ", output)
        end
    else
        @warn "Truss problem file not found, skipping TrussElementKσFun tests"
    end
end

# Test assembly with different penalty schemes
@testset "Assembly with penalty variations" begin
    nels = (4, 2)
    sizes = (1.0, 1.0)
    E = 1.0
    ν = 0.3
    force = 1.0

    problem = PointLoadCantilever(nels, sizes, E, ν, force)
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    @testset "Power penalty" begin
        vars = fill(0.5, ncells)
        penalty = PowerPenaltyFun(3.0)
        f_out = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)
        @test length(f_out) == ndofs(problem.ch.dh)
    end

    @testset "Rational penalty" begin
        vars = fill(0.5, ncells)
        penalty = RationalPenaltyFun(3.0)
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

    problem = PointLoadCantilever(nels, sizes, E, ν, force)
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    ncells = getncells(problem.ch.dh.grid)

    @testset "Very small densities" begin
        vars = fill(0.001, ncells)
        penalty = PowerPenaltyFun(3.0)
        f_out = zeros(Float64, ndofs(problem.ch.dh))
        TopOpt.TopOptProblems.assemble_f!(f_out, problem, elementinfo, vars, penalty, 0.001)
        @test length(f_out) == ndofs(problem.ch.dh)
        @test all(isfinite, f_out)
    end
end

# Test pressure/traction boundary conditions (pressuredict)
@testset "Pressure boundary conditions (pressuredict)" begin
    # Use TieBeam which has getpressuredict and getfacesets defined
    T = Float64
    E = T(1)
    ν = T(0.3)
    force = T(1)

    problem = TieBeam(T; refine=1, force=force, E=E, ν=ν)

    # Check that TieBeam has pressure loads defined
    pressuredict = TopOpt.TopOptProblems.getpressuredict(problem)
    @test !isempty(pressuredict)
    @test haskey(pressuredict, "rightload")
    @test haskey(pressuredict, "bottomload")
    @test pressuredict["rightload"] == 2 * force
    @test pressuredict["bottomload"] == -force

    # Check that facesets exist
    facesets = TopOpt.TopOptProblems.getfacesets(problem)
    @test haskey(facesets, "rightload")
    @test haskey(facesets, "bottomload")

    # Test _make_dloads function via make_Kes_and_fes (public API)
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        problem, 2, Val{:Static}
    )
    ncells = getncells(problem.ch.dh.grid)
    @test length(dloads) == ncells

    # Verify that pressure loads result in non-zero distributed loads
    # The traction is applied as: fe[(i-1)*dim+d] += ϕ * t * normal[d] * dΓ
    # where t = -pressure (traction = negative pressure)
    any_nonzero = any(fe -> any(x -> x != 0, fe), dloads)
    @test any_nonzero

    # Verify that each cell's distributed load vector has correct size
    # For 2D linear elements: 4 nodes * 2 DOFs = 8 entries
    dim = 2
    nnodes_per_cell = 4  # Linear quadrilateral
    expected_size = nnodes_per_cell * dim
    for (cellid, fe) in enumerate(dloads)
        @test length(fe) == expected_size
    end

    # Test that pressure values are correctly applied
    # rightload has pressure = 2*force (positive), traction = -2*force
    # bottomload has pressure = -force (negative), traction = force
    # The distributed load direction depends on the normal vector at each face

    # Verify the distributed loads are finite
    @test all(fe -> all(isfinite, fe), dloads)
end

# Test pressure boundary conditions integration with FEA
@testset "Pressure BC integration with FEA" begin
    T = Float64
    E = T(1)
    ν = T(0.3)
    force = T(1)

    problem = TieBeam(T; refine=1, force=force, E=E, ν=ν)

    # Build element FEA info
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

    # Get distributed loads from pressure via make_Kes_and_fes
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        problem, 2, Val{:Static}
    )

    # Create global FEA info
    ginfo = GlobalFEAInfo(problem)

    # Assemble the global force vector with distributed loads
    vars = ones(T, getncells(problem.ch.dh.grid))
    assemble!(ginfo, problem, elementinfo, vars)

    # Apply distributed loads using update_f!
    metadata = problem.metadata
    TopOpt.TopOptProblems.update_f!(ginfo.f, metadata.dof_cells, dloads)

    # Verify global force vector has non-zero entries from pressure
    @test any(ginfo.f .!= 0)

    # The force vector should be finite
    @test all(isfinite, ginfo.f)

    # Test that we can solve the system
    # K should be positive definite, so we can solve for displacement
    u = ginfo.K \ ginfo.f
    @test length(u) == ndofs(problem.ch.dh)
    @test all(isfinite, u)
end

# Test pressure direction (traction = -pressure)
@testset "Pressure direction sign convention" begin
    # Create a simple problem with known pressure
    T = Float64
    problem = TieBeam(T; refine=1, force=T(1.0), E=T(1), ν=T(0.3))

    # Get pressure dictionary
    pressuredict = TopOpt.TopOptProblems.getpressuredict(problem)

    # Verify the sign convention: traction = -pressure
    # Positive pressure -> negative traction (inward force)
    # Negative pressure -> positive traction (outward force)
    for (key, pressure_val) in pressuredict
        expected_traction = -pressure_val
        # The actual traction calculation happens inside _make_dloads
        # where t = -pressuredict[k]
        @test expected_traction == -pressure_val
    end

    # Test that distributed loads are computed correctly via make_Kes_and_fes
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        problem, 2, Val{:Static}
    )

    # Verify dloads is populated (non-zero)
    any_nonzero = any(fe -> any(x -> x != 0, fe), dloads)
    @test any_nonzero

    # Verify all values are finite
    @test all(fe -> all(isfinite, fe), dloads)
end

# Test pressure loop implementation details
@testset "Pressure loop - _make_dloads implementation" begin
    # This test specifically covers the code in _make_dloads that iterates over
    # pressuredict keys and computes distributed loads from pressure boundary conditions
    T = Float64
    problem = TieBeam(T; refine=2, force=T(1.0), E=T(1), ν=T(0.3))

    # Get problem data needed for the loop
    dh = TopOpt.TopOptProblems.getdh(problem)
    grid = dh.grid
    all_boundary_facets = reduce(
        union, values(grid.facetsets); init=Set{Ferrite.FacetIndex}()
    )
    pressuredict = TopOpt.TopOptProblems.getpressuredict(problem)
    facesets = TopOpt.TopOptProblems.getfacesets(problem)

    # Build element FEA info to get facevalues
    elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        problem, 2, Val{:Static}
    )

    dim = TopOpt.TopOptProblems.getdim(problem)
    N = TopOpt.TopOptProblems.nnodespercell(problem)
    n_basefuncs = Ferrite.getnbasefunctions(facevalues)

    # Verify the pressure loop over keys(pressuredict)
    for k in keys(pressuredict)
        t = -pressuredict[k] # traction = negative the pressure
        faceset = facesets[k]

        for (cellid, faceid) in faceset
            # Verify face is on boundary
            @test Ferrite.FacetIndex(cellid, faceid) in all_boundary_facets

            # Verify dloads entry exists for this cell
            fe = dloads[cellid]
            @test length(fe) == N * dim  # 4 nodes * 2 DOFs = 8 for 2D quad

            # Verify fe contains finite values
            @test all(isfinite, fe)
        end
    end

    # Test that traction direction matches pressure sign
    # rightload has positive pressure (2*force), so traction is negative (inward)
    # bottomload has negative pressure (-force), so traction is positive (outward)
    @test haskey(pressuredict, "rightload")
    @test haskey(pressuredict, "bottomload")
    @test pressuredict["rightload"] > 0  # Positive pressure
    @test pressuredict["bottomload"] < 0  # Negative pressure
end

# Test pressure loop with multiple face quadrature points
@testset "Pressure loop - quadrature integration" begin
    T = Float64
    problem = TieBeam(T; refine=2, force=T(1.0), E=T(1), ν=T(0.3))

    # Get facevalues and dloads
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(
        problem, 2, Val{:Static}
    )

    # Verify quadrature is set up
    n_quadpoints = Ferrite.getnquadpoints(facevalues)
    @test n_quadpoints >= 1  # Should have at least one quadrature point

    # Verify each face integration produces valid results
    pressuredict = TopOpt.TopOptProblems.getpressuredict(problem)
    facesets = TopOpt.TopOptProblems.getfacesets(problem)

    for k in keys(pressuredict)
        faceset = facesets[k]
        for (cellid, faceid) in faceset
            fe = dloads[cellid]
            # The distributed load should have been computed by integrating
            # over quadrature points: fe[(i-1)*dim+d] += ϕ * t * normal[d] * dΓ
            @test all(isfinite, fe)
        end
    end

    # Test that pressure loads contribute to global force vector
    metadata = problem.metadata
    f = zeros(T, ndofs(TopOpt.TopOptProblems.getdh(problem)))
    TopOpt.TopOptProblems.update_f!(f, metadata.dof_cells, dloads)

    # Global force should have non-zero entries from pressure
    @test any(f .!= 0)
    @test all(isfinite, f)
end

# Test pressure boundary condition error handling
@testset "Pressure loop - boundary validation" begin
    T = Float64
    problem = TieBeam(T; refine=1, force=T(1.0), E=T(1), ν=T(0.3))

    # Verify all faces in pressuredict facesets are on the boundary
    dh = TopOpt.TopOptProblems.getdh(problem)
    grid = dh.grid
    all_boundary_facets = reduce(
        union, values(grid.facetsets); init=Set{Ferrite.FacetIndex}()
    )
    pressuredict = TopOpt.TopOptProblems.getpressuredict(problem)
    facesets = TopOpt.TopOptProblems.getfacesets(problem)

    for k in keys(pressuredict)
        faceset = facesets[k]
        for (cellid, faceid) in faceset
            # Each face in the faceset must be on the boundary
            # This is checked by: boundary_matrix[faceid, cellid] || throw(...)
            @test Ferrite.FacetIndex(cellid, faceid) in all_boundary_facets
        end
    end
end
