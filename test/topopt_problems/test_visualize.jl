using TopOpt
using Makie
using CairoMakie
using Test

@testset "Continuum visualize smoke tests" begin
    # Node-based Dirichlet BCs (cantilever) and facet-based BCs (LBeam,
    # TieBeam) must both render: support drawing expands FacetIndex sets to
    # their facet nodes.
    cantilever = PointLoadCantilever(Val{:Linear}, (4, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    lbeam = LBeam(Val{:Linear}; length=6, height=6, upperslab=3, lowerslab=3)
    lbeam_dist = LBeam(
        Val{:Linear}; length=6, height=6, upperslab=3, lowerslab=3, load_width=3
    )

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
