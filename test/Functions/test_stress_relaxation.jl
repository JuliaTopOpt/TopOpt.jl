using TopOpt, Zygote, FiniteDifferences, LinearAlgebra, Test, Random
using StatsFuns: logsumexp
const FDM = FiniteDifferences
using TopOpt.Functions: epsilon_relaxed

Random.seed!(1)

@testset "epsilon_relaxed" begin
    N = 10
    σv = rand(N) .+ 0.5
    ρ = rand(N)
    σlim = 1.0

    @testset "limit behavior" begin
        # Zero density always satisfies the relaxed constraint
        @test epsilon_relaxed([1e3], [0.0], σlim, 0.01)[] == -0.01
        # ε = 0 with ρ = 1 recovers the unrelaxed constraint residual
        σ = 1.2
        @test epsilon_relaxed([σ], [1.0], σlim, 0.0)[] ≈ σ / σlim - 1
    end

    @testset "values" begin
        ε = 0.1
        g = epsilon_relaxed(σv, ρ, σlim, ε)
        @test g ≈ @.(ρ * (σv / σlim - 1) - ε)
        # satisfied in the void limit even with huge stress
        @test all(epsilon_relaxed(1e6 .* ones(N), zeros(N), σlim, ε) .< 0)
    end

    @testset "gradient vs finite differences" begin
        ε = 0.1
        # KS-aggregated ε-relaxed constraints, as used in the tutorials
        f = σ -> logsumexp(10.0 .* epsilon_relaxed(σ, ρ, σlim, ε)) / 10.0
        fd_grad = FiniteDifferences.grad(FDM.central_fdm(5, 1), f, σv)[1]
        @test Zygote.gradient(f, σv)[1] ≈ fd_grad rtol = 1e-6
    end
end

@testset "Relaxed von Mises stress function" begin
    nels = (3, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    xmin = 0.01
    solver = FEASolver(DirectSolver, problem; xmin=xmin, penalty=PowerPenaltyFun(3.0))

    σf_micro = von_mises_stress_function(solver)
    α = 0.5
    σf_relaxed = von_mises_stress_function(solver; stress_exponent=α)

    x = clamp.(rand(prod(nels)), 0.0, 1.0)

    @testset "relaxed stress equals ρ^α times microscopic stress" begin
        σ_micro = σf_micro(PseudoDensities(x))
        σ_relaxed = σf_relaxed(PseudoDensities(x))
        ρ = @. x * (1 - xmin) + xmin
        @test σ_relaxed ≈ ρ .^ α .* σ_micro
        # Relaxed stress vanishes in void elements, microscopic does not
        x_void = zeros(prod(nels))
        σ_relaxed_void = σf_relaxed(PseudoDensities(x_void))
        σ_micro_void = σf_micro(PseudoDensities(x_void))
        @test all(σ_relaxed_void .≈ (xmin^α) .* σ_micro_void)
    end

    @testset "stress_exponent = 0 reproduces the microscopic stress" begin
        σf0 = von_mises_stress_function(solver; stress_exponent=0)
        @test σf0(PseudoDensities(x)) == σf_micro(PseudoDensities(x))
    end

    @testset "gradient of aggregated relaxed stress vs finite differences" begin
        x0 = clamp.(rand(prod(nels)), 0.2, 1.0)
        N = length(x0)
        # Normalized p-norm of the relaxed stress, as used in the tutorials
        f = x -> norm(σf_relaxed(PseudoDensities(x)), 8) / N^(1 / 8)
        fd_grad = FiniteDifferences.grad(FDM.central_fdm(5, 1), f, x0)[1]
        @test Zygote.gradient(f, x0)[1] ≈ fd_grad rtol = 1e-4
    end

    @testset "finite gradient at zero density" begin
        # The xmin density floor keeps the relaxation factor differentiable
        x0 = zeros(prod(nels))
        f = x -> sum(σf_relaxed(PseudoDensities(x)))
        g = Zygote.gradient(f, x0)[1]
        @test all(isfinite, g)
    end
end
