using TopOpt, Nonconvex, Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays
const FDM = FiniteDifferences

Random.seed!(1)

@testset "Compliance" begin
    nels = (10, 10)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        comp = Compliance(problem, solver)
        for i in 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            val1, grad1 = Nonconvex.value_gradient(comp, x)
            val2, grad2 = comp(x), Zygote.gradient(comp, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), comp, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "Displacement" begin
    nels = (10, 10)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        dp = Displacement(solver)
        u = dp(solver.vars)
        for _ in 1:3
             x = clamp.(rand(prod(nels)), 0.1, 1.0)
             v = rand(length(u))
             f = x -> dot(dp(x), v)
             val1, grad1 = Nonconvex.value_gradient(f, x)
             val2, grad2 = f(x), Zygote.gradient(f, x)[1]
             grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
             @test val1 == val2
             @test norm(grad1 - grad2) == 0
             @test norm(grad2 - grad3) <= 1e-4
        end
    end
end

@testset "Volume" begin
    nels = (10, 10)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        vol = Volume(problem, solver)
        constr = x -> vol(x) - 0.3
        for i in 1:3
            x = rand(prod(nels))
            val1, grad1 = Nonconvex.value_gradient(constr, x)
            val2, grad2 = constr(x), Zygote.gradient(constr, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), constr, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "DensityFilter" begin
    nels = (10, 10)
    problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        filter = TopOpt.DensityFilter(solver, rmin = 4.0)
        for i in 1:3
            x = rand(prod(nels))
            v = rand(prod(nels))
            f = FunctionWrapper(x -> dot(filter(x), v), 1)
            val1, grad1 = Nonconvex.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-5
        end
    end
end

@testset "SensFilter" begin
    nels = (10, 10)
    problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0))
    solver = FEASolver(
        Direct, problem, xmin = 1e-3, penalty = PowerPenalty(3.0),
    )
    sensfilter = SensFilter(solver, rmin = 4.0)
    x = rand(length(solver.vars))
    y = rand(length(x))
    @test sensfilter(x) == x
    grad = Zygote.gradient(x -> dot(sensfilter(x), y), x)[1]
    @test grad != y
end

@testset "Block compliance" begin
    nels = (10, 10)
    nloads = 10
    base_problem = PointLoadCantilever(Val{:Linear}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    dense_load_inds = vec(TopOpt.TopOptProblems.get_surface_dofs(base_problem))
    dense_rank = 3
    F = spzeros(TopOpt.Ferrite.ndofs(base_problem.ch.dh), nloads)
    Fsize = size(F)
    for i in 1:dense_rank
        F += sparsevec(dense_load_inds, randn(length(dense_load_inds)) / dense_rank, Fsize[1]) * randn(Fsize[2])'
    end
    problem = MultiLoad(base_problem, F)
    for p in (1.0, 2.0, 3.0)
        solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(p))
        exact_svd_block = BlockCompliance(problem, solver, method = :exact)
        constr = Nonconvex.FunctionWrapper(x -> exact_svd_block(x) .- 1000.0, length(exact_svd_block(solver.vars)))
        for i in 1:3
            x = clamp.(rand(prod(nels)), 0.1, 1.0)
            v = rand(nloads)
            f = FunctionWrapper(x -> dot(constr(x), v), 1)
            val1, grad1 = Nonconvex.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            @test val1 == val2
            @test norm(grad1 - grad2) == 0
            @test norm(grad2 - grad3) <= 1e-4
        end
    end
end

@testset "Local stress" begin
    nels = (10, 10)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    for F in (MacroVonMisesStress, MicroVonMisesStress)
        for p in (1.0, 2.0, 3.0)
            solver = FEASolver(Direct, problem, xmin = 0.001, penalty = TopOpt.PowerPenalty(p))
            stress = F(solver)
            for i in 1:3
                x = clamp.(rand(prod(nels)), 0.1, 1.0)
                v = rand(prod(nels))
                f = FunctionWrapper(x -> dot(stress(x), v), 1)
                val1, grad1 = Nonconvex.value_gradient(f, x)
                val2, grad2 = f(x), Zygote.gradient(f, x)[1]
                grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
                @test val1 == val2
                @test norm(grad1 - grad2) == 0
                @test norm(grad1 - grad3) <= 1e-5
            end
        end
    end

    nels = (10, 10, 2)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0, 1.0), 1.0, 0.3, 1.0)
    for F in (MacroVonMisesStress, MicroVonMisesStress)
        for p in (1.0, 2.0, 3.0)
            solver = FEASolver(Direct, problem, xmin = 0.001, penalty = TopOpt.PowerPenalty(p))
            stress = F(solver)
            for i in 1:3
                x = clamp.(rand(prod(nels)), 0.1, 1.0)
                v = rand(prod(nels))
                f = FunctionWrapper(x -> dot(stress(x), v), 1)
                val1, grad1 = Nonconvex.value_gradient(f, x)
                val2, grad2 = f(x), Zygote.gradient(f, x)[1]
                grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
                @test val1 == val2
                @test norm(grad1 - grad2) == 0
                @test norm(grad1 - grad3) <= 1e-5
            end
        end
    end
end

@testset "AssembleK" begin
    nels = (2, 2)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    ak = AssembleK(problem)
    dh = problem.ch.dh
    total_ndof = ndofs(dh)
    T = eltype(problem.E)

    for _ in 1:3
        v = rand(T, total_ndof)
        f = Kx -> sum(ak(Kx)*v)
        Kes = [rand(T,k,k) for _ in 1:N]
        val1, grad1 = Nonconvex.value_gradient(f, Kes);
        val2, grad2 = f(Kes), Zygote.gradient(f, Kes)[1];
        grad3 = FDM.grad(central_fdm(5, 1), f, Kes)[1];
        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        @test norm(grad2 - grad3) <= 1e-4
    end
end
