using TopOpt, Test, LinearAlgebra, Zygote, ForwardDiff
using TopOpt: PowerPenalty, RationalPenalty, SinhPenalty, HeavisideProjection, SigmoidProjection, ProjectedPenalty, PseudoDensities, setpenalty!

@testset "PowerPenalty" begin
    # Test construction
    pp = PowerPenalty(3.0)
    @test pp.p == 3.0

    # Test application to scalar values
    @test pp(0.5) ≈ 0.5^3
    @test pp(0.0) ≈ 0.0
    @test pp(1.0) ≈ 1.0

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = pp(x)
    @test result.x ≈ [0.1^3, 0.5^3, 0.9^3]
    @test result isa PseudoDensities

    # Test derivative with ForwardDiff
    f(x) = PowerPenalty(3.0)(x)
    @test ForwardDiff.derivative(f, 0.5) ≈ 3.0 * 0.5^2
    @test ForwardDiff.derivative(f, 0.0) ≈ 0.0

    # Test derivative with Zygote
    g = x -> PowerPenalty(2.0)(x)
    @test Zygote.gradient(g, 0.5)[1] ≈ 2.0 * 0.5

    # Test copy
    pp_copy = copy(pp)
    @test pp_copy.p == pp.p
    pp_copy.p = 2.0
    @test pp.p == 3.0  # Original unchanged

    # Test setpenalty!
    pp2 = PowerPenalty(2.0)
    setpenalty!(pp2, 3.5)
    @test pp2.p == 3.5

    # Test different penalty values
    @test PowerPenalty(1.0)(0.5) ≈ 0.5
    @test PowerPenalty(2.0)(0.5) ≈ 0.25
    @test PowerPenalty(3.0)(0.5) ≈ 0.125
end

@testset "RationalPenalty" begin
    # Test construction
    rp = RationalPenalty(3.0)
    @test rp.p == 3.0

    # Test application to scalar values
    x = 0.5
    @test rp(x) ≈ x / (1.0 + rp.p * (1.0 - x))
    @test rp(0.0) ≈ 0.0
    @test rp(1.0) ≈ 1.0

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = rp(x)
    expected = [rp(0.1), rp(0.5), rp(0.9)]
    @test result.x ≈ expected

    # Test derivative with ForwardDiff
    f(x) = RationalPenalty(3.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Analytical derivative: (1 + p) / (1 + p(1-x))^2
    expected_df = (1.0 + 3.0) / (1.0 + 3.0 * (1.0 - 0.5))^2
    @test df ≈ expected_df

    # Test copy
    rp_copy = copy(rp)
    @test rp_copy.p == rp.p

    # Test setpenalty!
    setpenalty!(rp, 4.0)
    @test rp.p == 4.0
end

@testset "SinhPenalty" begin
    # Test construction
    sp = SinhPenalty(3.0)
    @test sp.p == 3.0

    # Test application to scalar values
    x = 0.5
    # sinh_penalty = sinh(p*x) / sinh(p)
    @test sp(x) ≈ sinh(3.0 * x) / sinh(3.0)
    @test sp(0.0) ≈ 0.0
    @test isapprox(sp(1.0), 1.0, atol=1e-10)

    # Test application to PseudoDensities
    x = PseudoDensities([0.1, 0.5, 0.9])
    result = sp(x)
    expected = [sp(0.1), sp(0.5), sp(0.9)]
    @test result.x ≈ expected

    # Test derivative with ForwardDiff
    f(x) = SinhPenalty(3.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Analytical derivative: p * cosh(p*x) / sinh(p)
    expected_df = 3.0 * cosh(3.0 * 0.5) / sinh(3.0)
    @test df ≈ expected_df

    # Test copy and setpenalty!
    sp_copy = copy(sp)
    @test sp_copy.p == sp.p
    setpenalty!(sp, 2.0)
    @test sp.p == 2.0
end

@testset "HeavisideProjection" begin
    # Test construction
    hp = HeavisideProjection(5.0)
    @test hp.β == 5.0

    # Test projection function
    x = 0.5
    # heaviside_proj = 1 - exp(-β*x) + x*exp(-β)
    expected = 1 - exp(-5.0 * x) + x * exp(-5.0)
    @test hp(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = hp(x_pd)
    @test result.x ≈ [hp(0.1), hp(0.5), hp(0.9)]

    # Test with AbstractArray
    x_arr = [0.1, 0.5, 0.9]
    result_arr = hp(x_arr)
    @test result_arr ≈ [hp(0.1), hp(0.5), hp(0.9)]

    # Test derivative with ForwardDiff
    f(x) = HeavisideProjection(5.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # At x = 0.5, derivative should be relatively large
    @test df > 0.0

    # Test at boundaries
    @test hp(0.0) < 0.01  # Should be close to 0
    @test hp(1.0) > 0.99  # Should be close to 1

    # Test copy
    hp_copy = copy(hp)
    @test hp_copy.β == hp.β

    # Test with different β values
    hp2 = HeavisideProjection(3.0)
    @test hp2.β == 3.0
    @test hp2(0.5) ≈ 1 - exp(-3.0 * 0.5) + 0.5 * exp(-3.0)
end

@testset "SigmoidProjection" begin
    # Test construction
    sp = SigmoidProjection(4.0)
    @test sp.β == 4.0

    # Test projection function
    x = 0.5
    # sigmoid_proj = 1 / (1 + exp((β+1)*(-x+0.5)))
    expected = 1.0 / (1.0 + exp((4.0 + 1) * (-x + 0.5)))
    @test sp(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = sp(x_pd)
    @test result.x ≈ [sp(0.1), sp(0.5), sp(0.9)]

    # Test with AbstractArray
    x_arr = [0.2, 0.5, 0.8]
    result_arr = sp(x_arr)
    @test result_arr ≈ [sp(0.2), sp(0.5), sp(0.8)]

    # Test derivative with ForwardDiff
    f(x) = SigmoidProjection(4.0)(x)
    df = ForwardDiff.derivative(f, 0.5)
    # Just verify derivative is positive and finite
    @test df > 0.0
    @test isfinite(df)

    # Test at boundaries (with β=4, sigmoid doesn't saturate completely)
    @test sp(0.0) < 0.1  # Should be relatively small
    @test sp(1.0) > 0.9  # Should be relatively large

    # Test copy
    sp_copy = copy(sp)
    @test sp_copy.β == sp.β

    # Test with different β values
    sp2 = SigmoidProjection(2.0)
    @test sp2.β == 2.0
    @test sp2(0.5) ≈ 1 / (1 + exp((2.0 + 1) * (-0.5 + 0.5)))
end

@testset "ProjectedPenalty" begin
    # Test construction
    pp = PowerPenalty(3.0)
    proj = HeavisideProjection(5.0)
    pp_proj = ProjectedPenalty(pp, proj)
    
    @test pp_proj.penalty == pp
    @test pp_proj.proj == proj

    # Test application: penalty(proj(x))
    x = 0.5
    expected = pp(proj(x))
    @test pp_proj(x) ≈ expected

    # Test with PseudoDensities
    x_pd = PseudoDensities([0.1, 0.5, 0.9])
    result = pp_proj(x_pd)
    expected_values = [pp(proj(0.1)), pp(proj(0.5)), pp(proj(0.9))]
    @test result.x ≈ expected_values

    # Test property forwarding
    @test pp_proj.p == 3.0  # Forwarded from PowerPenalty
    
    # Test copy
    pp_proj_copy = copy(pp_proj)
    @test pp_proj_copy.p == pp_proj.p
    @test pp_proj_copy.proj.β == proj.β

    # Test setpenalty!
    setpenalty!(pp_proj, 4.0)
    @test pp_proj.p == 4.0

    # Test with different penalty/projection combinations
    rp = RationalPenalty(2.0)
    sp = SigmoidProjection(3.0)
    rp_sp = ProjectedPenalty(rp, sp)
    
    x = PseudoDensities([0.2, 0.6, 0.8])
    result = rp_sp(x)
    expected = [rp(sp(0.2)), rp(sp(0.6)), rp(sp(0.8))]
    @test result.x ≈ expected

    # Test gradient
    f(x) = ProjectedPenalty(PowerPenalty(2.0), HeavisideProjection(3.0))(x)
    g = Zygote.gradient(f, 0.5)[1]
    @test g isa Real
end