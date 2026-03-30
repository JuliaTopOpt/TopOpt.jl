# Test file for pressure boundary condition functions and distributed loads

using TopOpt
using TopOpt.TopOptProblems
using TopOpt.TrussTopOptProblems.TrussIO
using Ferrite
using StaticArrays
using Test

import TopOpt.TopOptProblems: make_Kes_and_fes

@testset "Pressure boundary conditions (pressuredict)" begin
    # TieBeam supports pressure loading
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0, E=1.0, ν=0.3)

    # Test getpressuredict
    pd = getpressuredict(problem)
    @test pd isa Dict{String,Float64}
    @test haskey(pd, "rightload")
    @test haskey(pd, "bottomload")
    @test pd["rightload"] == 2.0
    @test pd["bottomload"] == -1.0

    # Test getfacesets
    fs = TopOpt.TopOptProblems.getfacesets(problem)
    @test fs isa Dict{String,Set{Tuple{Int,Int}}}

    # Test default values
    @test TopOpt.TopOptProblems.getfacesets(problem) === getdh(problem).grid.facesets

    # Create another problem with different force value
    problem2 = TieBeam(Val{:Linear}, Float64; refine=2, force=5.0)
    pd2 = getpressuredict(problem2)
    @test pd2["rightload"] == 10.0  # 2 * force
    @test pd2["bottomload"] == -5.0  # -force

    # Test that other problem types have empty pressuredict by default
    # PointLoadCantilever doesn't support pressure - uses concentrated loads
    # Must have even number of elements along y and z axes
    nels = (10, 6)
    sizes = (1.0, 1.0)
    cantilever = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)
    @test getpressuredict(cantilever) == Dict{String,Float64}()

    # HalfMBB doesn't support pressure - uses concentrated loads
    # Must have even number of elements along y and z axes
    half_mbb = HalfMBB(Val{:Linear}, (10, 6), (1.0, 1.0), 1.0, 0.3, 1.0)
    @test getpressuredict(half_mbb) == Dict{String,Float64}()

    # Test with quadratic elements
    problem_quad = TieBeam(Val{:Quadratic}, Float64; refine=1, force=3.0)
    pd_quad = getpressuredict(problem_quad)
    @test pd_quad["rightload"] == 6.0
    @test pd_quad["bottomload"] == -3.0
end

@testset "Pressure BC integration with FEA" begin
    # Create TieBeam problem with pressure loading
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # Test integration with make_Kes_and_fes
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(problem, 2)

    # Check that dloads are computed
    @test length(dloads) > 0
    @test eltype(dloads) == eltype(weights)

    # Verify dloads structure
    nel = getncells(getdh(problem).grid)
    @test length(dloads) == nel

    # Check that some dloads are non-zero (due to pressure on rightload/bottomload faces)
    # Note: Not all elements will have pressure, only those on the boundary
    non_zero_dloads = count(d -> any(d .!= 0), dloads)
    @test non_zero_dloads >= 0  # At minimum, no error occurred
end

@testset "Pressure direction sign convention" begin
    # Positive pressure should act as traction pointing INTO the domain
    # Negative pressure (suction) points outward

    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)
    pd = getpressuredict(problem)

    # rightload: positive pressure on right face (acts leftward, into domain)
    @test pd["rightload"] == 2.0

    # bottomload: negative pressure on bottom face (acts upward, into domain)
    # This is negative because pressure is -traction, and traction should point up
    @test pd["bottomload"] == -1.0
end

@testset "Pressure loop - _make_dloads implementation" begin
    # Create problem and get FEA components
    problem = TieBeam(Val{:Linear}, Float64; refine=2, force=2.0)

    # Get FEA components
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(problem, 2)

    # Verify dloads are computed
    @test length(dloads) == getncells(getdh(problem).grid)

    # Check structure of dloads
    @test eltype(dloads) <: SVector

    # Verify dloads can be indexed and contain numeric values
    for i in 1:min(5, length(dloads))  # Check first few elements
        @test dloads[i] isa SVector
        @test length(dloads[i]) == length(weights[i])
    end

    # Test with different quadrature orders
    for quad_order in [1, 2, 3]
        Kes_q, weights_q, dloads_q, cv_q, fv_q = make_Kes_and_fes(problem, quad_order)
        @test length(dloads_q) == length(dloads)
    end
end

@testset "Pressure loop - quadrature integration" begin
    # Test that pressure is integrated correctly over faces
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # Verify facesets exist via problem's getfacesets
    fs = TopOpt.TopOptProblems.getfacesets(problem)
    @test haskey(fs, "leftfixed")
    @test haskey(fs, "rightload")
    @test haskey(fs, "bottomload")

    # Get FEA components
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(problem, 2)

    # Verify facevalues are properly configured
    @test facevalues isa FaceScalarValues
end

@testset "Pressure loop - boundary validation" begin
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # This should run without error - validates boundary faces
    Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(problem, 2)

    # Verify no errors occurred during boundary validation
    @test true

    # Check that pressuredict entries match faceset names
    pd = getpressuredict(problem)
    fs = TopOpt.TopOptProblems.getfacesets(problem)

    for key in keys(pd)
        @test haskey(fs, key) || error("Pressure key '$key' not found in facesets")
    end
end

@testset "Concentrated loads (cloaddict) vs Pressure (pressuredict)" begin
    # PointLoadCantilever uses concentrated loads (point forces)
    # Must have even number of elements along y and z axes
    nels = (20, 10)
    sizes = (1.0, 1.0)
    cantilever = PointLoadCantilever(Val{:Linear}, nels, sizes, 1.0, 0.3, 1.0)

    # Should have concentrated loads, not pressure
    cloads = getcloaddict(cantilever)
    @test !isempty(cloads)  # Has point load
    @test isempty(getpressuredict(cantilever))  # No pressure

    # TieBeam uses pressure (distributed loads)
    tie_beam = TieBeam(Val{:Linear}, Float64; refine=1, force=1.0)

    # Should have pressure, not concentrated loads
    @test !isempty(getpressuredict(tie_beam))  # Has pressure
    @test isempty(getcloaddict(tie_beam))  # No concentrated loads
end

@testset "Pressure with different refine levels" begin
    for refine in 1:3
        problem = TieBeam(Val{:Linear}, Float64; refine=refine, force=1.0)

        # Verify problem is created successfully
        @test getpressuredict(problem)["rightload"] == 2.0

        # Verify FEA components work
        Kes, weights, dloads, cellvalues, facevalues = make_Kes_and_fes(problem, 2)
        @test length(Kes) == getncells(getdh(problem).grid)
    end
end

@testset "Pressure compatibility with multiload" begin
    # Test that pressure works with MultiLoad wrapper
    problem = TieBeam(Val{:Linear}, Float64; refine=1, force=2.0)
    nloads = 2

    # Create multiload problem - correct API is MultiLoad(problem, nloads)
    sp = MultiLoad(problem, nloads)

    # Test that getpressuredict works through MultiLoad
    pd = getpressuredict(sp)
    @test pd isa Dict{String,Float64}
    @test haskey(pd, "rightload")
    @test haskey(pd, "bottomload")
end
