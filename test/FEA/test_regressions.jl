# Regression tests for FEA correctness fixes:
#  - `FEA.update_f!` indexed the per-cell element matrices with a
#    prescribed-DOF index, raising a BoundsError whenever
#    `#prescribed_dofs > #cells` (e.g. a fully-clamped coarse mesh).
#  - A single-column matrix RHS fell through to the vector solve path and
#    raised a DimensionMismatch when the LHS was a vector.
using TopOpt, Test, LinearAlgebra, Random, SparseArrays
using Ferrite: getncells, ndofs, ConstraintHandler, Dirichlet, getfacetset, add!, close!, update!

Random.seed!(42)

@testset "update_f! with more prescribed DOFs than cells" begin
    # A (2,2) cantilever has 6 prescribed DOFs (left edge: 3 nodes x 2) but
    # only 4 cells, which previously crashed `update_f!` with a BoundsError.
    nels = (2, 2)
    problem = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(DirectSolver, problem; xmin=0.001)
    elementinfo = solver.elementinfo
    metadata = elementinfo.metadata
    Kes = elementinfo.Kes
    ch = problem.ch
    @test length(ch.prescribed_dofs) > length(Kes)

    f = zeros(eltype(solver.u), ndofs(problem.ch.dh))
    TopOpt.FEA.update_f!(
        f,
        ch.inhomogeneities,
        ch.prescribed_dofs,
        false,
        metadata.dof_cells,
        metadata.cell_dofs,
        Kes,
        solver.xmin,
        TopOpt.Utilities.getpenalty(solver),
        solver.vars,
        1.0,
    )
    @test all(isfinite, f)
end

@testset "Matrix-free solve with more prescribed DOFs than cells" begin
    # A fully-clamped 2x2 mesh has 8 prescribed DOFs (all 4 boundary nodes x
    # 2) but only 4 cells; the matrix-free path must still solve.
    nels = (2, 2)
    base = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    dh = base.ch.dh

    # Clamp all four edges
    ch = ConstraintHandler(dh)
    grid = dh.grid
    for setname in ("left", "right", "top", "bottom")
        add!(ch, Dirichlet(:u, getfacetset(grid, setname), (x, t) -> [0.0, 0.0], [1, 2]))
    end
    close!(ch)
    update!(ch, 0.0)
    problem = PointLoadCantilever(
        base.rect_grid, base.E, base.ν, ch, base.force, base.force_dof, base.metadata,
    )

    @test length(ch.prescribed_dofs) > getncells(problem)

    solver_mf = FEASolver(CGMatrixFreeSolver, problem; xmin=0.001)
    solver_direct = FEASolver(DirectSolver, problem; xmin=0.001)
    x = rand(getncells(problem))
    solver_mf.vars .= x
    solver_direct.vars .= x
    solver_mf()
    solver_direct()
    @test !any(isnan, solver_mf.u)
    @test solver_mf.u ≈ solver_direct.u rtol = 1e-6
end

@testset "Single-column matrix RHS with vector LHS" begin
    nels = (4, 4)
    base = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    F = spzeros(ndofs(base.ch.dh), 1)
    F[end, 1] = 1.0
    problem = MultiLoad(base, F)

    solver = FEASolver(DirectSolver, problem; xmin=0.01)
    x = fill(0.5, getncells(problem))
    solver.vars .= x

    # Matrix RHS with the default (vector) LHS: previously a DimensionMismatch
    solver(; rhs=Matrix(problem.F), assemble_f=false)
    @test !any(isnan, solver.lhs)
    @test !any(isinf, solver.lhs)

    # And with an explicit matrix LHS (the BlockComplianceFun usage pattern)
    Y = zeros(ndofs(base.ch.dh), 1)
    solver(; rhs=Matrix(problem.F), lhs=Y, assemble_f=false)
    @test all(isfinite, Y)
end

@testset "BlockComplianceFun exact method with a single scenario" begin
    nels = (4, 4)
    base = PointLoadCantilever(nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    F = spzeros(ndofs(base.ch.dh), 1)
    F[end, 1] = 1.0
    problem = MultiLoad(base, F)
    solver = FEASolver(DirectSolver, problem; xmin=0.01, penalty=PowerPenaltyFun(2.0))
    bc = BlockComplianceFun(problem, solver; method=:exact)
    x = fill(0.5, getncells(problem))
    result = bc(PseudoDensities(x))
    @test length(result) == 1
    @test isfinite(result[1])
    @test result[1] > 0
end
