using TopOpt, Test, LinearAlgebra, Random, FiniteDifferences, Zygote
using TopOpt: ndofs, Nonconvex
const FDM = FiniteDifferences

Random.seed!(42)

@testset "Thermal Compliance - Gradient Verification" begin
    @testset "Gradient matches finite differences" begin
        nels = (8, 6)
        sizes = (1.0, 1.0)
        k = 1.0
        # Apply heat flux on top boundary (faceset "top")
        heatflux = Dict("top" => 100.0)  # 100 W/m² into the domain
        problem = HeatConductionProblem(Val{:Linear}, nels, sizes, k; Tleft=0.0, Tright=0.0, heatflux=heatflux)

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(1.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))

        # Test at multiple random points
        for i in 1:3
            x = clamp.(rand(prod(nels)), 0.2, 1.0)

            val = f(x)
            grad_zygote = Zygote.gradient(f, x)[1]

            # Check value is reasonable
            @test val > 0
            @test isfinite(val)

            # Check gradient properties
            @test length(grad_zygote) == length(x)
            @test all(isfinite.(grad_zygote))

            # Compute finite difference gradient
            fd_grad = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]

            # Gradient should match finite differences
            @test isapprox(grad_zygote, fd_grad; rtol=1e-3, atol=1e-6)
        end
    end

    @testset "Gradient sign is correct" begin
        # Compliance gradient should be negative (more material = lower compliance)
        # This is because dJ/dx = -T^T Ke T * dρ/dx
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heatflux = Dict("top" => 1.0)

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)

        x = fill(0.5, prod(nels))
        grad = Zygote.gradient(x -> comp(PseudoDensities(x)), x)[1]

        # All gradients should be negative (more material = lower compliance)
        @test all(grad .< 0)
    end
end

@testset "Thermal Compliance - Physical Validation" begin
    @testset "Higher conductivity gives lower thermal compliance" begin
        nels = (8, 4)
        sizes = (1.0, 1.0)
        heatflux = Dict("top" => 1.0)

        # Compare two cases with different thermal conductivity
        k_low = 0.5
        k_high = 5.0

        problem_low = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k_low;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )
        problem_high = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k_high;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )

        solver_low = FEASolver(DirectSolver, problem_low; xmin=0.01)
        solver_high = FEASolver(DirectSolver, problem_high; xmin=0.01)

        comp_low = ThermalCompliance(solver_low)
        comp_high = ThermalCompliance(solver_high)

        x = ones(prod(nels))
        tc_low = comp_low(PseudoDensities(x))
        tc_high = comp_high(PseudoDensities(x))

        # Higher conductivity should give lower thermal compliance (better heat dissipation)
        @test tc_high < tc_low
    end

    @testset "Heat flux scaling" begin
        # Doubling heat flux should double thermal compliance
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0

        problem1 = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        problem2 = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 2.0)
        )

        solver1 = FEASolver(DirectSolver, problem1; xmin=0.01)
        solver2 = FEASolver(DirectSolver, problem2; xmin=0.01)

        comp1 = ThermalCompliance(solver1)
        comp2 = ThermalCompliance(solver2)

        x = ones(prod(nels))
        tc1 = comp1(PseudoDensities(x))
        tc2 = comp2(PseudoDensities(x))

        # Thermal compliance J = Q^T T
        # If Q doubles, T doubles (linear system), so J should be 4x
        # Actually: J = Q^T T, with T = K^{-1} Q, so J = Q^T K^{-1} Q
        # If Q -> 2Q, then J -> 4J
        @test isapprox(tc2, 4 * tc1; rtol=0.05)
    end
end

@testset "Heat Transfer - Error Handling" begin
    @testset "ThermalCompliance rejects structural problems" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError ThermalCompliance(solver)
    end

    @testset "Compliance rejects heat transfer problems" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, 1.0;
            Tleft=0.0, Tright=0.0
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError Compliance(solver)
    end
end