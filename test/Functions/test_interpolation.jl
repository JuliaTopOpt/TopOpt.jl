using TopOpt
using TopOpt.Utilities
using Test

# Test interpolation functions for SIMP and material models
@testset "Interpolation Tests" begin
    @testset "SIMP penalization" begin
        # Test standard SIMP interpolation using PowerPenalty from Utilities
        p = 3.0
        penalty = PowerPenalty(p)
        
        # Test PowerPenalty on scalar value
        x = 0.5
        result = penalty(x)
        expected = x^p
        @test result ≈ expected
        @test result > 0
        @test result <= 1.0
        
        # Test at bounds
        @test penalty(1.0) ≈ 1.0
        @test penalty(0.0) ≈ 0.0
        
        # Test with PseudoDensities
        xmin = 0.001
        ρs = [xmin, 0.3, 0.5, 0.7, 1.0]
        pd = PseudoDensities(ρs)
        result_pd = penalty(pd)
        @test length(result_pd.x) == length(ρs)
        @test all(result_pd.x .> 0)
        @test all(result_pd.x .<= 1.0)
    end

    @testset "RAMP interpolation" begin
        # Test RAMP (Rational Approximation of Material Properties) using RationalPenalty
        q = 3.0
        ramp_penalty = RationalPenalty(q)
        
        # Test RAMP function on scalar
        x = 0.5
        result = ramp_penalty(x)
        expected = x / (1 + q * (1 - x))
        @test result ≈ expected
        @test result > 0
        @test result <= 1.0
        
        # Test at bounds
        @test ramp_penalty(1.0) ≈ 1.0
        @test ramp_penalty(0.0) ≈ 0.0
    end

    @testset "Sinh penalty interpolation" begin
        # Test SinhPenalty
        p = 2.0
        sinh_penalty = SinhPenalty(p)
        
        # Test on scalar
        x = 0.5
        result = sinh_penalty(x)
        expected = sinh(p * x) / sinh(p)
        @test result ≈ expected
        @test result > 0
        @test result <= 1.0
        
        # Test at bounds
        @test sinh_penalty(1.0) ≈ 1.0
        @test sinh_penalty(0.0) ≈ 0.0
    end

    @testset "Projection functions" begin
        # Test HeavisideProjection
        β = 10.0
        proj = HeavisideProjection(β)
        
        # Test on scalar
        x = 0.5
        result = proj(x)
        expected = 1 - exp(-β * x) + x * exp(-β)
        @test result ≈ expected
        @test result > 0
        @test result <= 1.0
        
        # Test SigmoidProjection
        β = 5.0
        sigmoid_proj = SigmoidProjection(β)
        result2 = sigmoid_proj(x)
        expected2 = 1 / (1 + exp((β + 1) * (-x + 0.5)))
        @test result2 ≈ expected2
    end

    @testset "ProjectedPenalty" begin
        # Test combination of penalty and projection
        p = 3.0
        β = 10.0
        power_penalty = PowerPenalty(p)
        heaviside_proj = HeavisideProjection(β)
        proj_penalty = ProjectedPenalty(power_penalty, heaviside_proj)
        
        # Test on scalar
        x = 0.5
        result = proj_penalty(x)
        # Should apply projection then penalty
        @test result > 0
        @test result <= 1.0
        
        # Test at bounds
        @test proj_penalty(1.0) ≈ 1.0
        @test proj_penalty(0.0) ≈ 0.0
        
        # Test with PseudoDensities
        ρs = [0.0, 0.25, 0.5, 0.75, 1.0]
        pd = PseudoDensities(ρs)
        result_pd = proj_penalty(pd)
        @test length(result_pd.x) == length(ρs)
    end
end
