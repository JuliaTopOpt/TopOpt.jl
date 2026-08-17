using TopOpt
using Makie
using CairoMakie
using Test

@testset "Continuum visualize smoke tests" begin
    # Node-based Dirichlet BCs (cantilever) and facet-based BCs (LBeam,
    # TieBeam) must both render: support drawing expands FacetIndex sets to
    # their facet nodes.
    cantilever = PointLoadCantilever((4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    lbeam = LBeam(; length=6, height=6, upperslab=3, lowerslab=3)
    lbeam_dist = LBeam(; length=6, height=6, upperslab=3, lowerslab=3, load_width=3)

    for problem in (cantilever, lbeam, lbeam_dist)
        fig = visualize(
            problem;
            topology=ones(getncells(TopOptProblems.getdh(problem).grid)),
            default_exagg_scale=0.0,
        )
        @test fig isa Makie.Figure
    end

    # Non-finite topologies fail fast with a descriptive error instead of a
    # cryptic Cairo crash
    @test_throws ArgumentError visualize(
        lbeam; topology=[NaN; ones(getncells(TopOptProblems.getdh(lbeam).grid) - 1)]
    )
end

@testset "Non-interactive visualization (CairoMakie)" begin
    cantilever = PointLoadCantilever((4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    topology = ones(getncells(TopOptProblems.getdh(cantilever).grid))

    # interactive=false should produce a figure without sliders
    fig = visualize(cantilever; topology, interactive=false)
    @test fig isa Makie.Figure

    # Custom arrow colors and linewidths should be accepted without error
    fig = visualize(
        cantilever;
        topology,
        interactive=false,
        load_arrow_color=RGBAf(1.0, 0.0, 0.0, 1.0),
        support_arrow_color=RGBAf(0.0, 1.0, 0.0, 1.0),
        load_arrow_linewidth=3.0,
        support_arrow_linewidth=3.0,
        arrow_quality=30,
    )
    @test fig isa Makie.Figure

    # CairoMakie auto-detection disables interactive controls and still renders.
    fig = visualize(cantilever; topology, interactive=true)
    @test fig isa Makie.Figure
end

@testset "Static visualization keyword dispatch" begin
    cantilever = PointLoadCantilever((4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    topology = ones(getncells(TopOptProblems.getdh(cantilever).grid))

    # The static path requires WGLMakie; CairoMakie should reject it at the
    # backend boundary rather than forwarding `static` to mesh!.
    @test_throws ArgumentError visualize(cantilever; topology, static=true)
end

@testset "Level-set (OpenLSTO) visualization" begin
    L = TopOpt.OpenLSTO
    holes = L.LevelSetHole[
        L.LevelSetHole(8, 4, 2),
        L.LevelSetHole(16, 4, 2),
        L.LevelSetHole(24, 4, 2),
        L.LevelSetHole(12, 8, 2),
        L.LevelSetHole(20, 8, 2),
        L.LevelSetHole(28, 8, 2),
        L.LevelSetHole(8, 12, 2),
        L.LevelSetHole(16, 12, 2),
        L.LevelSetHole(24, 12, 2),
    ]
    result = L.compliance_minimization(;
        nelx=40, nely=20, holes=holes, max_iterations=2, verbose=false
    )

    fig = visualize(result)
    @test fig isa Makie.Figure

    fig = visualize(result; interactive=false)
    @test fig isa Makie.Figure

    # The static path requires WGLMakie; CairoMakie rejects it at the boundary.
    @test_throws ArgumentError visualize(result; static=true)
end

@testset "Level-set (OpenLSTO) 3D visualization" begin
    L = TopOpt.OpenLSTO
    r3 = L.compliance_minimization_3d(;
        nelx=8, nely=4, nelz=4, max_iterations=2, verbose=false
    )

    fig = visualize(r3.level_set)
    @test fig isa Makie.Figure

    fig = visualize(r3.level_set; interactive=false)
    @test fig isa Makie.Figure

    @test_throws ArgumentError visualize(r3.level_set; static=true)
end

@testset "Truss visualization with new options" begin
    using TopOpt.TrussTopOptProblems

    # Use an existing truss instance from the test suite
    ins_dir = joinpath(@__DIR__, "../truss_topopt_problems/instances/ground_meshes")
    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        joinpath(ins_dir, "tim_2d.json")
    )
    loads = load_cases["0"]
    truss_problem = TrussProblem(node_points, elements, loads, fixities, mats, crosssecs)
    topology = ones(getncells(truss_problem))

    # Non-interactive mode
    fig = visualize(truss_problem; topology, interactive=false)
    @test fig isa Makie.Figure

    # Custom arrow options
    fig = visualize(
        truss_problem;
        topology,
        interactive=false,
        load_arrow_color=RGBAf(0.0, 0.0, 1.0, 1.0),
        support_arrow_color=RGBAf(1.0, 1.0, 0.0, 1.0),
        load_arrow_linewidth=4.0,
        support_arrow_linewidth=4.0,
    )
    @test fig isa Makie.Figure
end
