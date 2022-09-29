using TopOpt,
    Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays, ForwardDiff
const FDM = FiniteDifferences
using TopOpt: ndofs
using Ferrite: ndofs_per_cell, getncells

Random.seed!(1)

@testset "Compliance" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        comp = Compliance(solver)
        f = x -> comp(PseudoDensities(x))
        for i = 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            val1, grad1 = NonconvexCore.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "Displacement" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        dp = Displacement(solver)
        u = dp(PseudoDensities(solver.vars))
        for _ = 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            v = rand(length(u))
            f = x -> dot(dp(PseudoDensities(x)), v)
            val1, grad1 = NonconvexCore.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-4
        end
    end
end

@testset "Volume" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        vol = Volume(solver)
        constr = x -> vol(PseudoDensities(x)) - 0.3
        for i = 1:3
            x = rand(prod(nels))
            val1, grad1 = NonconvexCore.value_gradient(constr, x)
            val2, grad2 = constr(x), Zygote.gradient(constr, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), constr, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "DensityFilter" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        filter = TopOpt.DensityFilter(solver; rmin = 4.0)
        for i = 1:3
            x = rand(prod(nels))
            v = rand(prod(nels))
            f = FunctionWrapper(x -> dot(filter(PseudoDensities(x)), v), 1)
            val1, grad1 = NonconvexCore.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "SensFilter" begin
    nels = (2, 2)
    problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0))
    solver = FEASolver(Direct, problem; xmin = 1e-3, penalty = PowerPenalty(3.0))
    sensfilter = SensFilter(solver; rmin = 4.0)
    f = x -> sensfilter(PseudoDensities(x))
    x = rand(length(solver.vars))
    y = rand(length(x))
    @test f(x) == x
    grad = Zygote.gradient(x -> dot(f(x), y), x)[1]
    @test grad != y
end

@testset "Block compliance" begin
    nels = (2, 2)
    nloads = 10
    base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
    dense_rank = 3
    F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
    Fsize = size(F)
    for i = 1:dense_rank
        F +=
            sparsevec(
                dense_load_inds,
                randn(length(dense_load_inds)) / dense_rank,
                Fsize[1],
            ) * randn(Fsize[2])'
    end
    problem = MultiLoad(base_problem, F)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        exact_svd_block = BlockCompliance(problem, solver; method = :exact)
        constr = Nonconvex.FunctionWrapper(
            x -> exact_svd_block(x) .- 1000.0,
            length(exact_svd_block(PseudoDensities(solver.vars))),
        )
        for i = 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            v = rand(nloads)
            f = FunctionWrapper(x -> dot(constr(PseudoDensities(x)), v), 1)
            val1, grad1 = NonconvexCore.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-4
        end
    end
end

@testset "Stress tensor" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        st = StressTensor(solver)
        # element stress tensor - element 1
        est = st[1]
        dp = Displacement(solver)
        for i = 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            u = dp(PseudoDensities(x))
            s = st(u)
            est(u)
            # test element stress tensors
            map(1:4) do i
                est = st[i]
                f = u -> vec(est(TopOpt.Functions.DisplacementResult(u)))
                j1 = FDM.jacobian(central_fdm(5, 1), f, u.u)[1]
                j2 = Zygote.jacobian(f, u)[1]
                @test norm(j1 - j2) < 1e-7
            end
            # test all stress tensors
            f = u -> reduce(vcat, vec.(st(TopOpt.Functions.DisplacementResult(u))))
            j1 = FDM.jacobian(central_fdm(5, 1), f, u.u)[1]
            j2 = Zygote.jacobian(f, u.u)[1]
            @test norm(j1 - j2) < 1e-7
        end
    end
end
