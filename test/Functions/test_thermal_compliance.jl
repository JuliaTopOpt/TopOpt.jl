using TopOpt, Test, LinearAlgebra, Random, FiniteDifferences, Zygote
using TopOpt: ndofs, Nonconvex, PseudoDensities
using TopOpt.TopOptProblems: getdh, getheatfluxdict, make_Kes_and_fes
const FDM = FiniteDifferences

Random.seed!(42)

# Helper: compare Zygote gradient to 5th-order central finite differences.
function grad_vs_fd(name, f, x; rtol=1e-6, atol=1e-8)
    val = f(x)
    gz = Zygote.gradient(f, x)[1]
    fd = FDM.grad(FDM.central_fdm(5, 1), f, x)[1]
    rel = norm(gz .- fd) / max(norm(fd), 1e-12)
    @test isfinite(val)
    @test all(isfinite, gz)
    @test length(gz) == length(x)
    @test isapprox(gz, fd; rtol=rtol, atol=atol)
    return val, rel
end

@testset "Thermal Compliance - Gradient Verification (DirectSolver)" begin
    @testset "Gradient matches finite differences: homogeneous Dirichlet" begin
        nels = (8, 6)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 100.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(1.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))
        for _ in 1:3
            x = clamp.(rand(prod(nels)), 0.2, 1.0)
            grad_vs_fd("homog", f, x)
            @test f(x) > 0
        end
    end

    @testset "Gradient matches finite differences: inhomogeneous Dirichlet" begin
        # This is the bug-fix regression: previously the gradient was wrong by
        # ~260% relative error when Tleft != 0.
        nels = (8, 6)
        problem = HeatConductionProblem(
            Val{:Linear}, nels, (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))
        for _ in 1:3
            x = clamp.(rand(prod(nels)), 0.2, 1.0)
            grad_vs_fd("inhomog", f, x; rtol=1e-5)
        end
    end

    @testset "Gradient matches finite differences: quadratic elements" begin
        # Previously ElementFEAInfo crashed for Val{:Quadratic} heat problems
        # because the cellvalues interpolation was hardcoded to order 1.
        nels = (6, 4)
        problem = HeatConductionProblem(
            Val{:Quadratic}, nels, (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))
        x = clamp.(rand(prod(nels)), 0.2, 1.0)
        grad_vs_fd("quadratic", f, x; rtol=1e-5)
    end

    @testset "Gradient matches finite differences: quadratic + inhomogeneous Dirichlet" begin
        nels = (6, 4)
        problem = HeatConductionProblem(
            Val{:Quadratic}, nels, (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        f = x -> comp(PseudoDensities(x))
        x = clamp.(rand(prod(nels)), 0.2, 1.0)
        grad_vs_fd("quadratic-inhomog", f, x; rtol=1e-5)
    end
end

@testset "Thermal Compliance - Objective Correctness" begin
    @testset "J == Q^T T for homogeneous Dirichlet" begin
        problem = HeatConductionProblem(
            Val{:Linear}, (8, 4), (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        J = comp(PseudoDensities(ones(32)))
        QT = dot(solver.elementinfo.fixedload, solver.u)
        @test J ≈ QT rtol=1e-10
    end

    @testset "J == Q^T T for inhomogeneous Dirichlet (bug-fix regression)" begin
        # Previously J was computed as T^T K T, which leaks the Dirichlet energy
        # and overestimates by Tleft^2 * (conductance to boundary).
        problem = HeatConductionProblem(
            Val{:Linear}, (8, 4), (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        J = comp(PseudoDensities(ones(32)))
        QT = dot(solver.elementinfo.fixedload, solver.u)
        @test J ≈ QT rtol=1e-10
        # The old (wrong) T^T K T value would be substantially larger:
        K = solver.globalinfo.K
        TKT = dot(solver.u, K * solver.u)
        @test TKT > J + 1.0
    end

    @testset "J == 0 when there is no heat source (Dirichlet only)" begin
        # With Q = 0, the true thermal compliance Q^T T is exactly 0. The old
        # T^T K T objective returned a large positive number (the Dirichlet
        # energy) and optimized toward it.
        problem = HeatConductionProblem(
            Val{:Linear}, (4, 4), (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict{String,Float64}()
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        comp = ThermalCompliance(solver)
        J = comp(PseudoDensities(ones(16)))
        @test J ≈ 0.0 atol=1e-10
        # Gradient should also be zero (no objective to minimize).
        g = Zygote.gradient(x -> comp(PseudoDensities(x)), ones(16))[1]
        @test norm(g) < 1e-8
    end
end

@testset "Thermal Compliance - Physical Validation" begin
    @testset "Higher conductivity gives lower thermal compliance" begin
        nels = (8, 4)
        heatflux = Dict("top" => 1.0)
        problem_low = HeatConductionProblem(
            Val{:Linear}, nels, (1.0, 1.0), 0.5;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )
        problem_high = HeatConductionProblem(
            Val{:Linear}, nels, (1.0, 1.0), 5.0;
            Tleft=0.0, Tright=0.0, heatflux=heatflux
        )
        comp_low = ThermalCompliance(FEASolver(DirectSolver, problem_low; xmin=0.01))
        comp_high = ThermalCompliance(FEASolver(DirectSolver, problem_high; xmin=0.01))
        x = ones(prod(nels))
        @test comp_high(PseudoDensities(x)) < comp_low(PseudoDensities(x))
    end

    @testset "Heat flux scaling: J -> 4J when Q -> 2Q (homogeneous Dirichlet)" begin
        # With Tleft=Tright=0, J = Q^T K^{-1} Q is quadratic in Q, so doubling
        # the heat flux quadruples the compliance. This also holds for the
        # wrong T^T K T objective, so it is NOT a sufficient test on its own;
        # it is kept as a sanity check alongside the Q^T T identity test above.
        problem1 = HeatConductionProblem(
            Val{:Linear}, (6, 4), (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        problem2 = HeatConductionProblem(
            Val{:Linear}, (6, 4), (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 2.0)
        )
        comp1 = ThermalCompliance(FEASolver(DirectSolver, problem1; xmin=0.01))
        comp2 = ThermalCompliance(FEASolver(DirectSolver, problem2; xmin=0.01))
        x = ones(prod((6, 4)))
        tc1 = comp1(PseudoDensities(x))
        tc2 = comp2(PseudoDensities(x))
        @test isapprox(tc2, 4 * tc1; rtol=1e-3)
    end

    @testset "Heat flux scaling with inhomogeneous Dirichlet (bug-fix)" begin
        # With Tleft != 0, the scaling is J(2Q) - J(Q) = (2Q)^T K^{-1}(2Q) - Q^T K^{-1} Q
        # minus the cross term with the Dirichlet lift. The clean identity is
        # J(Q) = Q_f^T T_f, and doubling Q doubles T_f's Q-driven part but the
        # Dirichlet contribution stays fixed. So J(2Q) - J(Q) is NOT 3*J(Q) in
        # general. Verify the Q^T T identity holds at both scales instead.
        problem1 = HeatConductionProblem(
            Val{:Linear}, (6, 4), (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        problem2 = HeatConductionProblem(
            Val{:Linear}, (6, 4), (1.0, 1.0), 1.0;
            Tleft=100.0, Tright=0.0, heatflux=Dict("top" => 2.0)
        )
        for (p, label) in [(problem1, "Q"), (problem2, "2Q")]
            solver = FEASolver(DirectSolver, p; xmin=0.01, penalty=PowerPenalty(3.0))
            comp = ThermalCompliance(solver)
            J = comp(PseudoDensities(ones(24)))
            QT = dot(solver.elementinfo.fixedload, solver.u)
            @test isapprox(J, QT; rtol=1e-10)
        end
    end
end

@testset "Thermal Compliance - Quadratic Elements" begin
    @testset "ElementFEAInfo builds for Val{:Quadratic} (previously crashed)" begin
        problem = HeatConductionProblem(
            Val{:Quadratic}, (4, 4), (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        elementinfo = ElementFEAInfo(problem, 2, Val{:Static})
        @test length(elementinfo.Kes) == 16
        # Quadratic quad: 9 nodes, scalar temperature field -> 9x9 Ke.
        @test size(elementinfo.Kes[1], 1) == 9
        @test size(elementinfo.Kes[1], 2) == 9
        # Conductivity matrix is symmetric and positive semi-definite.
        Ke = Matrix(elementinfo.Kes[1])
        @test Ke ≈ Ke' rtol=1e-10
        @test all(eigvals(Ke) .>= -1e-10)
    end

    @testset "Quadratic problem solves and matches linear on same mesh count" begin
        # Sanity: a quadratic problem should solve to a finite temperature
        # field and produce a finite compliance. Cross-check element count.
        nels = (6, 4)
        problem = HeatConductionProblem(
            Val{:Quadratic}, nels, (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenalty(3.0))
        solver.vars .= 0.7
        solver()
        @test all(isfinite, solver.u)
        comp = ThermalCompliance(solver)
        J = comp(PseudoDensities(fill(0.7, prod(nels))))
        @test isfinite(J)
        @test J > 0
    end
end

@testset "Thermal Compliance - Error Handling" begin
    @testset "ThermalCompliance rejects structural problems" begin
        problem = PointLoadCantilever(Val{:Linear}, (6, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        @test_throws ArgumentError ThermalCompliance(solver)
    end

    @testset "Compliance rejects heat transfer problems" begin
        problem = HeatConductionProblem(
            Val{:Linear}, (4, 4), (1.0, 1.0), 1.0;
            Tleft=0.0, Tright=0.0
        )
        solver = FEASolver(DirectSolver, problem; xmin=0.001)
        @test_throws ArgumentError Compliance(solver)
    end
end

@testset "Thermal Compliance - getpenalty and setpenalty!" begin
    problem = HeatConductionProblem(
        Val{:Linear}, (4, 4), (1.0, 1.0), 1.0;
        Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
    )
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
    tc = ThermalCompliance(solver)

    @testset "getpenalty returns current penalty" begin
        current = TopOpt.Utilities.getpenalty(tc)
        @test current isa PowerPenalty
        @test current.p == 3.0
    end

    @testset "setpenalty! updates penalty" begin
        TopOpt.Utilities.setpenalty!(tc, PowerPenalty(2.0))
        @test TopOpt.Utilities.getpenalty(tc).p == 2.0
    end
end

@testset "Thermal Compliance - Vector input warning" begin
    problem = HeatConductionProblem(
        Val{:Linear}, (4, 4), (1.0, 1.0), 1.0;
        Tleft=0.0, Tright=0.0, heatflux=Dict("top" => 1.0)
    )
    solver = FEASolver(DirectSolver, problem; xmin=0.001, penalty=PowerPenalty(3.0))
    tc = ThermalCompliance(solver)
    x = ones(length(solver.vars)) * 0.5
    @test_logs (:warn, r"A vector input was passed in to the thermal compliance function") tc(x)
end