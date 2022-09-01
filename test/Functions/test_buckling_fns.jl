using TopOpt, Zygote, FiniteDifferences, LinearAlgebra, Test, Random, SparseArrays
const FDM = FiniteDifferences
using Ferrite: ndofs_per_cell, getncells
using TopOpt: ndofs
using Arpack
include("..//truss_topopt_problems//utils.jl")

gm_ins_dir = joinpath(@__DIR__, "..", "truss_topopt_problems", "instances", "ground_meshes");

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
        f = Kx -> sum(ak(Kx) * v)
        Kes = [rand(T, k, k) for _ in 1:N]
        Kes .= transpose.(Kes) .+ Kes
        val1, grad1 = NonconvexCore.value_gradient(f, Kes)
        val2, grad2 = f(Kes), Zygote.gradient(f, Kes)[1]
        grad3 = FDM.grad(central_fdm(5, 1), f, Kes)[1]
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
    solver = FEASolver(Direct, problem; xmin=0.01, penalty=TopOpt.PowerPenalty(1.0))

    ek = ElementK(solver)
    dh = problem.ch.dh
    T = eltype(problem.E)
    N = getncells(dh.grid)
    k = ndofs_per_cell(dh)

    # * check stiffness matrix consistency
    Kes_1 = ek(PseudoDensities(ones(T, prod(nels))))
    for (ci, (k1, k0)) in enumerate(zip(Kes_1, solver.elementinfo.Kes))
        @test k1 ≈ k0
    end

    for _ in 1:3
        vs = [rand(T, k, k) for i in 1:N]
        f = x -> begin
            Kes = ek(PseudoDensities(x))
            sum([sum(Kes[i] * vs[i]) for i in 1:length(x)])
        end

        x = clamp.(rand(prod(nels)), 0.1, 1.0)

        val1, grad1 = NonconvexCore.value_gradient(f, x)
        val2, grad2 = f(x), Zygote.gradient(f, x)[1]
        grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        @test norm(grad1 - grad3) <= 1e-5
    end
end

@testset "TrussElementKσ $problem_dim" for problem_dim in ["2d", "3d"]
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(gm_ins_dir, file_name)

    mats = TrussFEAMaterial(1.0, 0.3)
    crossecs = TrussFEACrossSec(800.0)

    node_points, elements, _, _, fixities, load_cases = load_truss_json(problem_file)
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs
    )
    solver = FEASolver(Direct, problem)
    solver()
    u = solver.u

    esigk = TrussElementKσ(problem, solver)
    nels = length(solver.vars)
    dh = problem.ch.dh
    T = eltype(u)
    N = getncells(dh.grid)
    k = ndofs_per_cell(dh)

    # * check geometric stiffness matrix consistency
    Kσs_0 = get_truss_Kσs(problem, u, solver.elementinfo.cellvalues)

    for _ in 1:3
        vs = [rand(T, k, k) for i in 1:N]
        f =
            x -> begin
                Keσs = esigk(TopOpt.Functions.DisplacementResult(u), PseudoDensities(x))
                sum([sum(Keσs[i] * vs[i]) for i in 1:length(x)])
            end

        x = clamp.(rand(nels), 0.1, 1.0)

        Kσs_1 = esigk(TopOpt.Functions.DisplacementResult(u), PseudoDensities(x))
        for (ci, (k1, k0)) in enumerate(zip(Kσs_1, Kσs_0))
            @test k1 ≈ k0 * x[ci]
        end

        val1, grad1 = NonconvexCore.value_gradient(f, x)
        val2, grad2 = f(x), Zygote.gradient(f, x)[1]
        grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
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
            return sum(M * v)
        end

        function f2(x)
            M = K * sum(x)
            M = apply_boundary_with_meandiag!(M, ch)
            return sum(M * v)
        end

        x = rand(total_ndof)
        for f in [f2] #f1, 
            val1, grad1 = NonconvexCore.value_gradient(f, x)
            val2, grad2 = f(x), Zygote.gradient(f, x)[1]
            grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]
            # @test val1 == val2
            # @test norm(grad1 - grad2) == 0
            @test norm(grad1 - grad3) <= 1e-5
        end
    end
end

@testset "Buckling SDP constraint $problem_dim" for problem_dim in ["2d", "3d"]
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(gm_ins_dir, file_name)

    mats = TrussFEAMaterial(1.0, 0.3)
    crossecs = TrussFEACrossSec(800.0)

    node_points, elements, _, _, fixities, load_cases = load_truss_json(problem_file)
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs
    )

    xmin = 0.0001 # minimum density
    p = 4.0 # penalty
    V = 0.5 # maximum volume fraction
    x0 = fill(1.0, ncells) # initial design

    solver = FEASolver(Direct, problem)
    ch = problem.ch
    dh = problem.ch.dh
    total_ndof = ndofs(dh)

    comp = TopOpt.Compliance(solver)
    dp = TopOpt.Displacement(solver)
    assemble_k = TopOpt.AssembleK(problem)
    element_k = ElementK(solver)
    truss_element_kσ = TrussElementKσ(problem, solver)

    # * comliance minimization objective
    obj = comp
    c = 1.0 # buckling load multiplier

    function buckling_matrix_constr(x)
        # * Array(K + c*Kσ) ⋟ 0, PSD
        # * solve for the displacement
        xd = PseudoDensities(x)
        u = dp(xd)

        # * x -> Kes, construct all the element stiffness matrices
        # a list of small matrices for each element (cell)
        Kes = element_k(xd)

        # * Kes -> K (global linear stiffness matrix)
        K = assemble_k(Kes)
        K = apply_boundary_with_meandiag!(K, ch)

        # * u_e, x_e -> Ksigma_e
        Kσs = truss_element_kσ(u, xd)

        # * Kσs -> Kσ
        Kσ = assemble_k(Kσs)
        Kσ = apply_boundary_with_zerodiag!(Kσ, ch)

        return Array(K + c * Kσ)
    end

    # * check initial design stability
    @test isfinite(logdet(cholesky(buckling_matrix_constr(x0))))

    for _ in 1:3
        v = rand(eltype(x0), total_ndof)
        f = x -> sum(buckling_matrix_constr(x) * v)

        x = clamp.(rand(ncells), 0.1, 1.0)

        solver.vars = x
        solver()
        K, G = buckling(problem, solver.globalinfo, solver.elementinfo, x; u=solver.u)
        @test K + G ≈ buckling_matrix_constr(x)

        val1, grad1 = NonconvexCore.value_gradient(f, x)
        val2, grad2 = f(x), Zygote.gradient(f, x)[1]
        grad3 = FDM.grad(central_fdm(5, 1), f, x)[1]

        @test val1 == val2
        @test norm(grad1 - grad2) == 0
        @test norm(grad1 - grad3) <= 1e-5
    end
end
