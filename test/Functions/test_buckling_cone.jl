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
    K, Kσ = TopOpt.buckling(solver)
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
    K, Kσ = TopOpt.buckling(solver)
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

@testset "BucklingElementCutsFun" begin
    problem = PointLoadCantilever((2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    λ̄ = 5.9
    m = getncells(problem)
    x = clamp.(rand(m) .* 0.9 .+ 0.1, 0.0, 1.0)
    x̂ = PseudoDensities(x)
    solver.vars .= x
    solver()
    K, Kσ = TopOpt.buckling(solver)
    ev = eigen(Symmetric(Matrix(Kσ.data)), Symmetric(Matrix(K.data)))
    V = [vec(ev.vectors[:, 1])]
    cfull = dot(solver.u, solver.globalinfo.f)
    bec = BucklingElementCutsFun(solver, λ̄, V; c̄=cfull)
    val = bec(x̂)
    na = length(bec.active)
    @test length(val) == 1 + na + 1
    # compliance component (shared u)
    @test val[1] ≈ dot(solver.u, solver.globalinfo.f) / cfull - 1 rtol=1e-12
    # element component matches ρ_e(K0e + λ̄G0e)
    u = copy(solver.u)
    dh = problem.ch.dh
    cv = Ferrite.CellCache(dh)
    gd = zeros(Int, Ferrite.ndofs_per_cell(dh))
    cellvalues = solver.elementinfo.cellvalues
    e1 = bec.active[1]
    Ferrite.celldofs!(gd, dh, e1)
    Ferrite.reinit!(cv, e1)
    Ferrite.reinit!(cellvalues, cv)
    G0e = TopOpt.TopOptProblems.element_kσ(problem, cellvalues, u[gd])
    K0e = Matrix(TopOpt.rawmatrix(solver.elementinfo.Kes[e1]))
    ρ_e, _ = TopOpt.Utilities.get_ρ_dρ(x[e1], TopOpt.Utilities.getpenalty(solver), solver.xmin)
    @test val[2] ≈ ρ_e .* (K0e + λ̄ .* G0e) rtol=1e-12
    # cut component matches BucklingCutFun
    cut = BucklingCutFun(solver, λ̄, V[1])
    @test val[end][1] ≈ cut(x̂) rtol=1e-12
    # FD gradients: compliance, one element-matrix entry, the cut
    g1 = Zygote.gradient(x -> bec(PseudoDensities(x))[1], x)[1]
    g_fd1 = FDM.grad(central_fdm(5, 1), x -> bec(PseudoDensities(x))[1], x)[1]
    @test norm(g1 - g_fd1) / max(norm(g_fd1), 1) < 1e-7
    g2 = Zygote.gradient(x -> bec(PseudoDensities(x))[2][1, 2], x)[1]
    g_fd2 = FDM.grad(central_fdm(5, 1), x -> bec(PseudoDensities(x))[2][1, 2], x)[1]
    @test norm(g2 - g_fd2) / max(norm(g_fd2), 1) < 1e-7
    g3 = Zygote.gradient(x -> bec(PseudoDensities(x))[end][1], x)[1]
    g_fd3 = FDM.grad(central_fdm(5, 1), x -> bec(PseudoDensities(x))[end][1], x)[1]
    @test norm(g3 - g_fd3) / max(norm(g_fd3), 1) < 1e-7

    # regression: with NO active elements (compliance-only bundle) the
    # compliance gradient must still be nonzero — it scatters over every
    # cell, not only the active ones.
    bec0 = BucklingElementCutsFun(solver, λ̄, Vector{Vector{Float64}}();
                                  active=Int[], c̄=cfull)
    g0 = Zygote.gradient(x -> bec0(PseudoDensities(x))[1], x)[1]
    g_fd0 = FDM.grad(central_fdm(5, 1), x -> bec0(PseudoDensities(x))[1], x)[1]
    @test norm(g0 - g_fd0) / max(norm(g_fd0), 1) < 1e-7
    @test !iszero(g0)
end

@testset "KSBucklingFun" begin
    problem = PointLoadCantilever((2, 2), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    m = getncells(problem)
    x = clamp.(rand(m) .* 0.9 .+ 0.1, 0.0, 1.0)
    x̂ = PseudoDensities(x)
    solver.vars .= x
    solver()
    K, Kσ = TopOpt.buckling(solver)
    # deterministic dense eigensolver: BLFs λ = -1/ν, K-normalized modes
    dense_eigs(Kd, Kσd, k) = begin
        ev = eigen(Symmetric(Matrix(Kσd)), Symmetric(Matrix(Kd)))
        perm = sortperm(ev.values)[1:min(k, length(ev.values))]
        λs = Float64[]
        Xs = Vector{Vector{Float64}}()
        for j in perm
            ν = ev.values[j]
            ν < -1.0e-14 || continue
            φ = vec(ev.vectors[:, j])
            push!(λs, -1.0 / ν)
            push!(Xs, φ ./ sqrt(dot(φ, Kd * φ)))
        end
        return λs, Xs
    end
    ksb = KSBucklingFun(solver, 2, 10.0; eigsolver=dense_eigs)
    g = Zygote.gradient(x -> ksb(PseudoDensities(x)), x)[1]
    g_fd = FDM.grad(central_fdm(5, 1), x -> ksb(PseudoDensities(x)), x)[1]
    @test norm(g - g_fd) / max(norm(g_fd), 1) < 1e-6
    # the value is the logsumexp lower bound of the smallest BLFs
    λs_dense = dense_eigs(K.data, Kσ.data, 2)[1]
    @test ksb(x̂) ≈ -(1 / 10.0) * log(sum(exp(-10.0 * λ) for λ in λs_dense)) rtol=1e-2
end
