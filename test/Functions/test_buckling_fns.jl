using TopOpt, Nonconvex, Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays
const FDM = FiniteDifferences
using TopOpt: ndofs
using Ferrite: ndofs_per_cell, getncells

@testset "AssembleK" begin
    nels = (2, 2)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    ak = AssembleK(problem)
    dh = problem.ch.dh
    total_ndof = ndofs(dh)
    T = eltype(problem.E)
    einfo = ElementFEAInfo(problem)
    k = size(einfo.Kes[1], 1)
    N = length(einfo.Kes)
    for _ in 1:3
        v = rand(T, total_ndof)
        f = Kx -> sum(ak(Kx)*v)
        Kes = [rand(T,k,k) for _ in 1:N]
        Kes .= transpose.(Kes) .+ Kes
        val1, grad1 = Nonconvex.value_gradient(f, Kes);
        val2, grad2 = f(Kes), Zygote.gradient(f, Kes)[1];
        grad3 = FDM.grad(central_fdm(5, 1), f, Kes)[1];
        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        map(1:length(grad2)) do i
            g1 = grad2[i]
            _g2 = grad3[i]
            g2 = (_g2' + _g2) / 2
            @test norm(g1 - g2) <= 1e-4
        end
    end
end

@testset "ElementK" begin
    nels = (2, 2)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    solver = FEASolver(Direct, problem, xmin = 0.01, penalty = TopOpt.PowerPenalty(1.0))

    ek = ElementK(solver)
    dh = problem.ch.dh
    T = eltype(problem.E)
    N = getncells(dh.grid)
    k = ndofs_per_cell(dh)

    for _ in 1:3
        vs = [rand(T,k,k) for i in 1:N]
        f = x -> begin 
            Kes = ek(x)
            sum([sum(Kes[i]*vs[i]) for i in 1:length(x)])
        end
        x = clamp.(rand(prod(nels)), 0.1, 1.0)

        val1, grad1 = Nonconvex.value_gradient(f, x);
        val2, grad2 = f(x), Zygote.gradient(f, x)[1];
        grad3 = FDM.grad(central_fdm(5, 1), f, x)[1];
        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        @test norm(grad1 - grad3) <= 1e-5
    end
end

@testset "TrussElementKσ" begin
    ins_dir = joinpath(@__DIR__, "..", "truss_topopt_problems", "instances", "ground_meshes");
    file_name = "tim_2d.geo"
    problem_file = joinpath(ins_dir, file_name)
    node_points, elements, fixities, load_cases = load_truss_geo(problem_file)
    loads = load_cases[1]
    mat = TrussFEAMaterial(1.0, 0.3);
    crossec = TrussFEACrossSec(800.0);

    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mat, crossec);
    solver = FEASolver(Direct, problem);
    solver()
    u = solver.u

    esigk = TrussElementKσ(problem, solver)
    nels = length(solver.vars)
    dh = problem.ch.dh
    T = eltype(u)
    N = getncells(dh.grid)
    k = ndofs_per_cell(dh)

    for _ in 1:3
        vs = [rand(T,k,k) for i in 1:N]
        f = x -> begin 
            Keσs = esigk(u, x)
            sum([sum(Keσs[i]*vs[i]) for i in 1:length(x)])
        end

        x = clamp.(rand(nels), 0.1, 1.0)
        val1, grad1 = Nonconvex.value_gradient(f, x);
        val2, grad2 = f(x), Zygote.gradient(f, x)[1];
        grad3 = FDM.grad(central_fdm(5, 1), f, x)[1];
        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        @test norm(grad1 - grad3) <= 1e-5
    end
end

@testset "apply_boundary" begin
    nels = (2, 2)
    problem = PointLoadCantilever(Val{:Quadratic}, nels, (1.0, 1.0), 1.0, 0.3, 1.0)
    ch = problem.ch
    dh = problem.ch.dh
    T = eltype(problem.E)
    total_ndof = ndofs(dh)

    for _ in 1:3
        v = rand(T, total_ndof)
        K = sprand(Float64, total_ndof, total_ndof, 0.75)
        K = K + K'

        function f1(x)
            M = K * sum(x)
            M = apply_boundary_with_zerodiag!(M, ch)
            return sum(M*v)
        end

        function f2(x)
            M = K * sum(x)
            M = apply_boundary_with_meandiag!(M, ch)
            return sum(M*v)
        end

        x = rand(total_ndof)
        for f in [f2] #f1, 
            val1, grad1 = Nonconvex.value_gradient(f, x);
            val2, grad2 = f(x), Zygote.gradient(f, x)[1];
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1];
            # @test val1 == val2
            # @test norm(grad1 - grad2) == 0
            @test norm(grad1 - grad3) <= 1e-5
        end
    end
end