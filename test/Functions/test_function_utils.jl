using TopOpt, Test, Random, ForwardDiff
using TopOpt.Utilities: get_ρ, get_ρ_dρ
using TopOpt.Utilities: PowerPenalty, RationalPenalty, SinhPenalty

Random.seed!(42)

@testset "Function Utilities - get_ρ and get_ρ_dρ" begin
    @testset "get_ρ basic functionality" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.01
        
        # Test at various density values
        for x_e in [0.1, 0.5, 0.9, 1.0]
            ρ = get_ρ(x_e, penalty, xmin)
            
            # Result should be a real number
            @test isa(ρ, Real)
            @test isfinite(ρ)
            @test ρ >= 0
        end
    end

    @testset "get_ρ_dρ returns value and gradient" begin
        penalty = PowerPenalty(2.0)
        xmin = 0.001
        
        for x_e in [0.1, 0.5, 1.0]
            val, grad = get_ρ_dρ(x_e, penalty, xmin)
            
            # Check types
            @test isa(val, Real)
            @test isa(grad, Real)
            @test isfinite(val)
            @test isfinite(grad)
            
            # Gradient should be approximately correct (compare with finite differences)
            ε = 1e-8
            val_plus = get_ρ(x_e + ε, penalty, xmin)
            val_minus = get_ρ(x_e - ε, penalty, xmin)
            fd_grad = (val_plus - val_minus) / (2ε)
            
            @test isapprox(grad, fd_grad; rtol=1e-5)
        end
    end

    @testset "get_ρ with different penalty types" begin
        xmin = 0.01
        x_e = 0.5
        
        for penalty in [PowerPenalty(3.0), RationalPenalty(3.0), SinhPenalty(3.0)]
            ρ = get_ρ(x_e, penalty, xmin)
            @test isa(ρ, Real)
            @test isfinite(ρ)
            @test ρ >= 0
            
            val, grad = get_ρ_dρ(x_e, penalty, xmin)
            @test isa(grad, Real)
            @test isfinite(grad)
        end
    end

    @testset "get_ρ boundary values" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.001
        
        # At x_e = xmin, should get penalized minimum density
        ρ_min = get_ρ(xmin, penalty, xmin)
        @test isfinite(ρ_min)
        @test ρ_min > 0
        
        # At x_e = 1.0, should get penalized full density
        ρ_max = get_ρ(1.0, penalty, xmin)
        @test isapprox(ρ_max, 1.0; atol=1e-6)
    end

    @testset "get_ρ_dρ consistency" begin
        # The value returned by get_ρ_dρ should match get_ρ
        penalty = PowerPenalty(3.0)
        xmin = 0.01
        
        for x_e in [0.1, 0.5, 0.9]
            val1 = get_ρ(x_e, penalty, xmin)
            val2, _ = get_ρ_dρ(x_e, penalty, xmin)
            @test isapprox(val1, val2; rtol=1e-10)
        end
    end

    @testset "get_ρ gradient properties" begin
        penalty = PowerPenalty(3.0)
        xmin = 0.01
        
        # For power penalty, gradient should be positive (more density = more stiffness)
        for x_e in [0.2, 0.5, 0.8]
            _, grad = get_ρ_dρ(x_e, penalty, xmin)
            @test grad > 0
        end
    end
end