using TopOpt,
    Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays, ForwardDiff
const FDM = FiniteDifferences
using TopOpt: ndofs
using Ferrite: ndofs_per_cell, getncells

Random.seed!(1)

get_pen_T(::PseudoDensities{<:Any,T,<:Any}) where {T} = T

@testset "Projections and penalties" begin
    for proj in (HeavisideProjection(5.0), SigmoidProjection(4.0))
        for T1 in (true, false), T2 in (true, false), T3 in (true, false)
            x = PseudoDensities{T1,T2,T3}(rand(4))
            @test typeof(proj(x)) === typeof(x)
            @test typeof(proj.(x)) === typeof(x)
            Zygote.gradient(sum ∘ proj ∘ PseudoDensities, x)
        end
    end

    for pen in (PowerPenalty(3.0), RationalPenalty(3.0), SinhPenalty(3.0))
        for T1 in (true, false), T2 in (true, false), T3 in (true, false)
            x = PseudoDensities{T1,T2,T3}(rand(4))
            @test get_pen_T(pen(x)) === true
            @test get_pen_T(ProjectedPenalty(pen)(x)) === true
            Zygote.gradient(sum ∘ pen ∘ PseudoDensities, x)
        end
    end
end

@testset "Compliance" begin
    nels = (2, 2)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        comp = Compliance(solver)
        f = x -> comp(PseudoDensities(x))
        for i in 1:3
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
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        dp = Displacement(solver)
        u = dp(PseudoDensities(solver.vars))
        for _ in 1:3
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
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        vol = Volume(solver)
        constr = x -> vol(PseudoDensities(x)) - 0.3
        for i in 1:3
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
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        filter = DensityFilter(solver; rmin=4.0)
        for i in 1:3
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
    solver = FEASolver(Direct, problem; xmin=1e-3, penalty=PowerPenalty(3.0))
    sensfilter = SensFilter(solver; rmin=4.0)
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
    for i in 1:dense_rank
        F +=
            sparsevec(
                dense_load_inds, randn(length(dense_load_inds)) / dense_rank, Fsize[1]
            ) * randn(Fsize[2])'
    end
    problem = MultiLoad(base_problem, F)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        exact_svd_block = BlockCompliance(problem, solver; method=:exact)
        constr = Nonconvex.FunctionWrapper(
            x -> exact_svd_block(x) .- 1000.0,
            length(exact_svd_block(PseudoDensities(solver.vars))),
        )
        for i in 1:3
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
        solver = FEASolver(Direct, problem; xmin=0.01, penalty=PowerPenalty(p))
        st = StressTensor(solver)
        # element stress tensor - element 1
        est = st[1]
        dp = Displacement(solver)
        for i in 1:3
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
