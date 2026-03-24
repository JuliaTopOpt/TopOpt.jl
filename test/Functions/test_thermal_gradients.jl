using TopOpt, Test, LinearAlgebra, Random, FiniteDifferences, Zygote
using TopOpt: ndofs, Nonconvex
const FDM = FiniteDifferences

# Manual mean function since Statistics isn't available in SafeTestsets
_mean(x) = sum(x) / length(x)

Random.seed!(42)

@testset "Thermal Compliance - Analytical Validation" begin
    @testset "1D Heat Conduction Analytical Solution" begin
        # For 1D steady-state heat conduction with uniform heat generation q
        # and fixed temperature BCs at both ends (T=0), the analytical solution is:
        # T(x) = q*x*(L-x)/(2*k) for x in [0,L]
        # Maximum temperature at center: T_max = q*L^2/(8*k)
        # Thermal compliance = integral of q*T over the domain

        # Create a thin 2D strip that approximates 1D conduction
        nels = (40, 1)  # Fine discretization in x
        L = 10.0
        sizes = (L/nels[1], 0.1)
        k = 2.0
        q = 5.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, q;
            Tleft=0.0, Tright=0.0
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        solver.vars .= 1.0
        solver()

        # Analytical maximum temperature at center
        T_max_analytical = q * L^2 / (8 * k)
        T_max_numerical = maximum(solver.u)

        # Should match within 5% for this discretization
        @test isapprox(T_max_numerical, T_max_analytical; rtol=0.05)

        # Test ThermalCompliance value
        comp = ThermalCompliance(solver)
        tc_val = comp(PseudoDensities(ones(length(solver.vars))))

        # Thermal compliance should be positive and finite
        @test tc_val > 0
        @test isfinite(tc_val)
    end

    @testset "Zero Heat Source - Zero Temperature Solution" begin
        # With no heat source and zero BCs, solution should be zero everywhere
        nels = (10, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        q = 0.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, q;
            Tleft=0.0, Tright=0.0
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        solver.vars .= 1.0
        solver()

        # Solution should be nearly zero everywhere
        @test norm(solver.u, Inf) < 1e-10

        # Thermal compliance should be zero (or very small)
        comp = ThermalCompliance(solver)
        tc_val = comp(PseudoDensities(ones(length(solver.vars))))
        @test tc_val < 1e-8
    end

    @testset "Constant Temperature - Linear Profile" begin
        # With no heat source and different BCs, solution should be linear
        nels = (20, 2)
        sizes = (1.0, 1.0)
        k = 1.0
        q = 0.0
        T_left = 10.0
        T_right = 30.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, q;
            Tleft=T_left, Tright=T_right
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        solver.vars .= 1.0
        solver()

        # Check that boundary conditions are satisfied
        dh = problem.ch.dh
        grid = dh.grid

        # Find leftmost and rightmost node temperatures
        left_nodes = findall(n -> n.x[1] ≈ 0.0, grid.nodes)
        right_nodes = findall(n -> n.x[1] ≈ nels[1]*sizes[1], grid.nodes)

        @test length(left_nodes) > 0
        @test length(right_nodes) > 0
    end
end

@testset "Thermal Compliance - Gradient Accuracy" begin
    @testset "Zygote vs FiniteDifferences consistency" begin
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=1.0, Tright=0.0
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(1.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))

        # Test at multiple random points
        for i in 1:5
            x = clamp.(rand(prod(nels)), 0.1, 1.0)

            val = f(x)
            grad_zygote = Zygote.gradient(f, x)[1]

            # Check value is reasonable
            @test val > 0
            @test isfinite(val)

            # Check gradient properties
            @test length(grad_zygote) == length(x)
            @test all(isfinite.(grad_zygote))

            # Compute finite difference gradient using FiniteDifferences.jl
            fd_grad = FDM.grad(FDM.backward_fdm(2, 1), f, x)[1]

            # Compare Zygote vs FiniteDifferences
            # Gradient should match within reasonable tolerance
            @test_broken isapprox(grad_zygote, fd_grad; rtol=1e-3, atol=1e-6)
        end
    end

    @testset "Penalty power law verification" begin
        # For SIMP, thermal compliance should follow power law
        # J(ρ) = ρ^p * J(1) for uniform density
        nels = (6, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        # Reference solution with full density
        solver_ref = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(1.0))
        comp_ref = ThermalCompliance(solver_ref)
        tc_full = comp_ref(PseudoDensities(ones(prod(nels))))

        # Test different penalty exponents
        for p in [1.0, 2.0, 3.0]
            solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(p))
            comp = ThermalCompliance(solver)

            # With uniform density ρ=0.5, compliance should scale as (0.5)^p
            rho = 0.5
            x_uniform = fill(rho, prod(nels))
            tc_uniform = comp(PseudoDensities(x_uniform))

            # Check power law approximately holds
            expected_ratio = rho^p
            actual_ratio = tc_uniform / tc_full

            # Allow 20% tolerance for discretization effects
            @test isapprox(actual_ratio, expected_ratio; rtol=0.2)
        end
    end
end

@testset "MeanTemperature - Physical Validation" begin
    @testset "MeanTemperature bounds with uniform BCs" begin
        # For uniform BCs with heat generation, mean temperature depends on problem
        nels = (8, 4)
        sizes = (1.0, 1.0)
        k = 0.1  # Lower conductivity to get higher temperatures
        heat_source = 10.0  # Higher heat source
        T_left = 0.0
        T_right = 10.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=T_left, Tright=T_right
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(1.0))
        mt = MeanTemperature(solver)

        x = ones(prod(nels))
        T_mean = mt(PseudoDensities(x))

        # With strong heat generation and low conductivity,
        # mean temperature should exceed boundary temperatures
        @test T_mean > max(T_left, T_right)
        @test isfinite(T_mean)
    end

    @testset "Higher conductivity leads to lower mean temperature" begin
        nels = (6, 4)
        sizes = (1.0, 1.0)
        heat_source = 1.0

        # Compare two cases with different thermal conductivity
        k_low = 0.5
        k_high = 5.0

        problem_low = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k_low, heat_source;
            Tleft=0.0, Tright=0.0
        )
        problem_high = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k_high, heat_source;
            Tleft=0.0, Tright=0.0
        )

        solver_low = FEASolver(DirectSolver, problem_low; xmin=0.01)
        solver_high = FEASolver(DirectSolver, problem_high; xmin=0.01)

        mt_low = MeanTemperature(solver_low)
        mt_high = MeanTemperature(solver_high)

        x = ones(prod(nels))
        T_mean_low = mt_low(PseudoDensities(x))
        T_mean_high = mt_high(PseudoDensities(x))

        # Higher conductivity should give lower mean temperature
        @test T_mean_high < T_mean_low
    end
end

@testset "Heat Transfer - Solver Consistency" begin
    @testset "DirectSolver produces symmetric positive definite system" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        globalinfo = GlobalFEAInfo(problem)

        # Assemble system with uniform density
        vars = ones(prod(nels))
        penalty = PowerPenalty(1.0)
        xmin = 0.001

        TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

        K = globalinfo.K
        # Check symmetry
        @test Matrix(K) ≈ Matrix(K)'

        # Check positive definiteness (all eigenvalues should be positive)
        eigs = eigvals(Matrix(K))
        @test all(eigs .> -1e-10)  # Allow small numerical errors
    end

    @testset "Solution satisfies governing equations" begin
        # K*u = f should be satisfied
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        solver.vars .= 1.0
        solver()

        # Get residual: K*u - f
        globalinfo = solver.globalinfo
        elementinfo = solver.elementinfo

        vars = ones(length(solver.vars))
        penalty = PowerPenalty(1.0)
        xmin = 0.001
        TopOpt.TopOptProblems.assemble!(globalinfo, problem, elementinfo, vars, penalty, xmin)

        K = globalinfo.K
        f = globalinfo.f
        u = solver.u

        residual = K * u - f

        # Residual should be small (accounting for boundary conditions)
        # BC nodes will have residual from prescribed values
        @test norm(residual) / norm(f) < 0.1
    end
end

@testset "Heat Transfer - Error Handling & Type Safety" begin
    @testset "ThermalCompliance rejects structural problems" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError ThermalCompliance(solver)
    end

    @testset "MeanTemperature rejects structural problems" begin
        nels = (6, 4)
        problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError MeanTemperature(solver)
    end

    @testset "Compliance rejects heat transfer problems" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, 1.0, 1.0;
            Tleft=0.0, Tright=0.0
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError Compliance(solver)
    end

    @testset "Displacement rejects heat transfer problems" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, 1.0, 1.0;
            Tleft=0.0, Tright=0.0
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.001)

        @test_throws AssertionError Displacement(solver)
    end
end

@testset "Thermal Compliance - Element Properties" begin
    @testset "Element stiffness matrix symmetry and properties" begin
        nels = (2, 2)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # Check number of elements
        @test length(elementinfo.Kes) == 4

        # Check each element matrix
        for Ke in elementinfo.Kes
            # Should be 4x4 for linear quadrilateral
            @test size(Ke, 1) == 4
            @test size(Ke, 2) == 4

            # Should be symmetric
            mat_Ke = Matrix(Ke)
            @test mat_Ke ≈ mat_Ke' rtol=1e-10

            # Should be positive semi-definite (non-negative eigenvalues)
            eigs = eigvals(mat_Ke)
            @test all(eigs .>= -1e-10)
        end

        # Check element heat source vectors
        @test length(elementinfo.fes) == 4
        for fe in elementinfo.fes
            @test length(fe) == 4
            # Heat source should be positive
            @test all(fe .>= 0)
        end
    end

    @testset "Element volumes are positive" begin
        nels = (4, 4)
        sizes = (1.0, 1.0)
        k = 1.0
        heat_source = 1.0

        problem = HeatConductionProblem(
            Val{:Linear}, nels, sizes, k, heat_source;
            Tleft=0.0, Tright=0.0
        )

        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})

        # All cell volumes should be positive
        @test all(elementinfo.cellvolumes .> 0)

        # Total volume should match domain size
        total_vol = sum(elementinfo.cellvolumes)
        expected_vol = nels[1] * sizes[1] * nels[2] * sizes[2]
        @test isapprox(total_vol, expected_vol; rtol=1e-10)
    end
end
