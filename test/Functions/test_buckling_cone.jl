using TopOpt, Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays
const FDM = FiniteDifferences

@testset "BucklingConeFun" begin
    problem = PointLoadCantilever((2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    λ̄ = 5.9
    bc = BucklingConeFun(solver, λ̄)

    x = rand(4)
    P = bc(PseudoDensities(x))
    @test P isa AbstractMatrix
    @test issymmetric(P)

    # The K part alone: P(λ̄=0) must be the condensed stiffness matrix —
    # checked physically via the equilibrium P0·u = f.
    bc0 = BucklingConeFun(solver, 0.0)
    P0 = bc0(PseudoDensities(x))
    @test norm(P0 * solver.u - solver.globalinfo.f) <=
          1e-10 * max(norm(solver.globalinfo.f), 1)

    # Adjoint check against finite differences: f(x) = ⟨V, P(x)⟩. The dense
    # path is used because ChainRules' 2-arg dense×sparse `dot` rrule is
    # unreliable; P is tiny in this test.
    V = rand(size(P))
    V = V + V'
    f = x -> sum(V .* Matrix(bc(PseudoDensities(x))))
    g_zyg = Zygote.gradient(f, x)[1]
    g_fd = FDM.grad(central_fdm(5, 1), f, x)[1]
    @test norm(g_zyg - g_fd) / max(norm(g_fd), 1) < 1e-6
end
