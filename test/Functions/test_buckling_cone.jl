using TopOpt, Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays
const FDM = FiniteDifferences
const Ferrite = TopOpt.Ferrite

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

@testset "BucklingCutFun" begin
    problem = PointLoadCantilever((2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    λ̄ = 5.9
    m = getncells(problem)
    dh = problem.ch.dh
    cellvalues = solver.elementinfo.cellvalues
    x = clamp.(rand(m) .* 0.9 .+ 0.1, 0.0, 1.0)
    x̂ = PseudoDensities(x)
    solver.vars .= x
    solver()
    K, Kσ = TopOpt.buckling(
        problem, solver.globalinfo, solver.elementinfo, solver.vars,
        TopOpt.Utilities.getpenalty(solver), solver.xmin,
    )
    ev = eigen(Symmetric(Matrix(Kσ.data)), Symmetric(Matrix(K.data)))
    v = vec(ev.vectors[:, 1])
    cut = BucklingCutFun(solver, λ̄, v)

    # The per-cell quadratic value must match the global-assembled
    # definition vᵀ(K + λ̄Kσ)v (exact for buckling modes, which vanish on
    # the prescribed DOFs).
    @test cut(x̂) ≈ dot(v, K, v) + λ̄ * dot(v, Kσ, v) rtol=1e-10

    # Precomputed-input forms: same value, frozen-state gradients.
    # `cut(x̂)` re-solved at x, so `solver.u` is the solution at x.
    u = copy(solver.u)
    cv = Ferrite.CellCache(dh)
    gd = zeros(Int, Ferrite.ndofs_per_cell(dh))
    Kσ0s = map(1:m) do ci
        Ferrite.celldofs!(gd, dh, ci)
        Ferrite.reinit!(cv, ci)
        Ferrite.reinit!(cellvalues, cv)
        TopOpt.TopOptProblems.element_kσ(problem, cellvalues, u[gd])
    end
    @test cut(x̂, u) ≈ cut(x̂) rtol=1e-12
    @test cut(x̂, u, Kσ0s) ≈ cut(x̂) rtol=1e-12

    # Self-contained form: full gradient (including the design dependence
    # of the displacement field) against finite differences.
    g1 = Zygote.gradient(x -> cut(PseudoDensities(x)), x)[1]
    g_fd = FDM.grad(central_fdm(5, 1), x -> cut(PseudoDensities(x)), x)[1]
    @test norm(g1 - g_fd) / max(norm(g_fd), 1) < 1e-7

    # FD reference with K re-assembled at the perturbed point (the
    # precomputed forms assume the solver state is current at x̂).
    f_ref = x -> begin
        solver.vars .= x
        solver()
        cut(PseudoDensities(x), u, Kσ0s)
    end
    solver.vars .= x
    solver()
    g_fd2 = FDM.grad(central_fdm(5, 1), f_ref, x)[1]
    g2 = Zygote.gradient(x -> cut(PseudoDensities(x), u), x)[1]
    g3 = Zygote.gradient(x -> cut(PseudoDensities(x), u, Kσ0s), x)[1]
    @test norm(g2 - g_fd2) / max(norm(g_fd2), 1) < 1e-7
    @test norm(g3 - g_fd2) / max(norm(g_fd2), 1) < 1e-7
end

@testset "BucklingCutsFun" begin
    problem = PointLoadCantilever((2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    λ̄ = 5.9
    m = getncells(problem)
    dh = problem.ch.dh
    cellvalues = solver.elementinfo.cellvalues
    x = clamp.(rand(m) .* 0.9 .+ 0.1, 0.0, 1.0)
    x̂ = PseudoDensities(x)
    solver.vars .= x
    solver()
    K, Kσ = TopOpt.buckling(
        problem, solver.globalinfo, solver.elementinfo, solver.vars,
        TopOpt.Utilities.getpenalty(solver), solver.xmin,
    )
    ev = eigen(Symmetric(Matrix(Kσ.data)), Symmetric(Matrix(K.data)))
    V = [vec(ev.vectors[:, 1]), 0.9 .* vec(ev.vectors[:, 1])]
    cuts = BucklingCutsFun(solver, λ̄, V)

    # The bundle value matches the individual cuts.
    vals = cuts(x)
    @test vals ≈ [
        BucklingCutFun(solver, λ̄, vi)(x̂) for vi in V
    ] rtol=1e-12
    # `cuts(x)` solved at x, so `solver.u` is the solution at x.
    u = copy(solver.u)
    cv = Ferrite.CellCache(dh)
    gd = zeros(Int, Ferrite.ndofs_per_cell(dh))
    Kσ0s = map(1:m) do ci
        Ferrite.celldofs!(gd, dh, ci)
        Ferrite.reinit!(cv, ci)
        Ferrite.reinit!(cellvalues, cv)
        TopOpt.TopOptProblems.element_kσ(problem, cellvalues, u[gd])
    end

    # Jacobian rows through the rrule (one adjoint solve per mode).
    _, back = Zygote.pullback(cuts, x)
    e1 = zeros(2)
    e1[1] = 1.0
    gv = back(e1)[1]
    gv_fd = FDM.grad(central_fdm(5, 1), x -> cuts(x)[1], x)[1]
    @test norm(gv - gv_fd) / max(norm(gv_fd), 1) < 1e-7

    # Precomputed-state form: frozen-state gradient.
    solver.vars .= x
    solver()
    gv3 = Zygote.gradient(x -> cuts(PseudoDensities(x), u, Kσ0s)[1], x)[1]
    gv3_fd = FDM.grad(central_fdm(5, 1), x -> begin
        solver.vars .= x
        solver()
        cuts(PseudoDensities(x), u, Kσ0s)[1]
    end, x)[1]
    @test norm(gv3 - gv3_fd) / max(norm(gv3_fd), 1) < 1e-7
end
