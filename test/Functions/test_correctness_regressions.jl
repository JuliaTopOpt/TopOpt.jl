# Regression tests for correctness fixes:
#  - `DisplacementFun` gradient with inhomogeneous (nonzero) Dirichlet BCs
#    must include the lift term (raw element matrices, not `bcmatrix`).
#  - `VolumeFun.project` must handle degenerate target volumes.
#  - Hadamard trace/diagonal estimation requires `nv == nrows`.
#  - Filter metadata must not contain duplicate nodes per cell.
using TopOpt, Test, LinearAlgebra, Random, SparseArrays, Zygote
using Ferrite:
    getncells, ndofs, ConstraintHandler, Dirichlet, getnodeset, add!, close!, update!
using FiniteDifferences
const FDM = FiniteDifferences

Random.seed!(42)

@testset "DisplacementFun gradient with inhomogeneous Dirichlet BCs" begin
    nels = (4, 4)
    problem = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    dh = problem.ch.dh

    # Rebuild the constraint handler with a nonzero prescribed displacement
    # on the fixed (left) edge. The lift term K_fp*u_p enters the derivative
    # only when the full element matrices are used.
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getnodeset(dh.grid, "fixed_all"), (x, t) -> [0.1, 0.0], [1, 2]))
    close!(ch)
    update!(ch, 0.0)
    problem = PointLoadCantilever(
        problem.rect_grid,
        problem.E,
        problem.ν,
        ch,
        problem.force,
        problem.force_dof,
        problem.metadata,
    )

    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(3.0))
    dp = DisplacementFun(solver)
    u = dp(PseudoDensities(solver.vars))
    # The prescribed displacement is visible at the fixed nodes
    fixed_nodes = getnodeset(dh.grid, "fixed_all")
    for n in fixed_nodes
        @test u[problem.metadata.node_dofs[1, n]] ≈ 0.1
    end

    for _ in 1:3
        x = clamp.(rand(prod(nels)), 0.1, 1.0)
        v = rand(length(u))
        f = x -> dot(dp(PseudoDensities(x)), v)
        grad_z = Zygote.gradient(f, x)[1]
        grad_fd = FDM.grad(central_fdm(5, 1), f, x)[1]
        @test norm(grad_z - grad_fd) <= 1e-4
    end
end

@testset "VolumeFun project degenerate targets" begin
    nels = (3, 3)
    problem = HalfMBB(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    vol = VolumeFun(solver; fraction=false)
    total_volume = sum(vol.cellvolumes)
    x = rand(9)

    # Target above the total volume: previously a BoundsError
    projected = TopOpt.Functions.project(vol, 1.5 * total_volume, x)
    @test all(projected .== 1)

    # Negative target: previously returned a single solid element
    projected = TopOpt.Functions.project(vol, -0.1, x)
    @test all(projected .== 0)

    # Fractional interface with fraction=true
    volf = VolumeFun(solver; fraction=true)
    @test all(TopOpt.Functions.project(volf, 1.5, x) .== 1)
    @test all(TopOpt.Functions.project(volf, -0.5, x) .== 0)
end

@testset "Hadamard sampling requires complete basis" begin
    n, m = 6, 3
    F = sprandn(n, m, 0.8)

    # nv < number of probe-matrix rows is silently biased for Hadamard
    # columns; fail fast instead.
    @test_throws ArgumentError TopOpt.Functions.TraceEstimationMean(
        F, 1, true, TopOpt.Functions.hadamard!
    )
    @test_throws ArgumentError TopOpt.Functions.DiagonalEstimation(F, 1, n)
    # nv == m is fine
    tem = TopOpt.Functions.TraceEstimationMean(F, m, true, TopOpt.Functions.hadamard!)
    @test size(tem.V) == (m, m)
    de = TopOpt.Functions.DiagonalEstimation(F, m, n)
    @test size(de.V) == (m, m)

    # End-to-end: the hadamard :trace estimator with nv == nloads matches the
    # exact mean compliance.
    base_problem = PointLoadCantilever((6, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
    nloads = 4
    nd = ndofs(base_problem.ch.dh)
    Fl = spzeros(nd, nloads)
    for i in 1:nloads
        # Right-edge DOFs (none prescribed for the cantilever)
        Fl[nd - 2 * i + 1, i] = 1.0
    end
    problem = MultiLoad(base_problem, Fl)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(2.0))
    x = fill(0.5, length(solver.vars))
    mc_exact = MeanComplianceFun(problem, solver; method=:exact)
    mc_had = MeanComplianceFun(
        problem, solver; method=:trace, nv=nloads, sample_method=:hadamard
    )
    @test mc_exact(PseudoDensities(x)) > 0
    @test mc_had(PseudoDensities(x)) ≈ mc_exact(PseudoDensities(x)) rtol = 1e-8
end

@testset "Filter neighbour lists are duplicate-free" begin
    problem = PointLoadCantilever((6, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    neighbours, weights = TopOpt.CheqFilters.get_neighbour_info(problem, 2.0)
    for (nodes, ws) in zip(neighbours, weights)
        @test length(unique(nodes)) == length(nodes)
        @test length(ws) == length(nodes)
        # Weight consistency: each node appears once with weight rmin - dist
        @test all(>=(0), ws)
    end
end

@testset "DensityFilterFun keeps raw density for isolated elements" begin
    # An element whose radius search finds no neighbouring nodes previously
    # got an all-zero filter row (its filtered density silently forced to
    # zero). The Jacobian must instead carry an identity row for it.
    problem = PointLoadCantilever((4, 4), (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    nel = getncells(problem)

    # Handcrafted metadata: the last element has no neighbours within rmin.
    neighs = Vector{Int}[[i] for i in 1:nel]
    weights = [[1.0] for i in 1:nel]
    neighs[end] = Int[]
    weights[end] = Float64[]
    meta = TopOpt.CheqFilters.FilterMetadata(neighs, weights)

    J = TopOpt.CheqFilters.getJacobian(solver, meta)
    @test size(J) == (nel, nel)
    # Every row is non-empty (no element is silently voided)
    @test all(i -> !iszero(J[i, :]), 1:nel)
    # The isolated element keeps its raw density
    @test J[end, end] == 1.0
    @test all(iszero, J[end, 1:(end - 1)])
end
