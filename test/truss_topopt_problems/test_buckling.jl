using Test
using LinearAlgebra
using Ferrite

using TopOpt
using TopOpt.TrussTopOptProblems: buckling
using Arpack
# using IterativeSolvers

fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples");
gm_ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

@testset "Tim buckling $problem_dim" for problem_dim in ["2d", "3d"]
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(gm_ins_dir, file_name)

    mats = TrussFEAMaterial(1.0, 0.3);
    crossecs = TrussFEACrossSec(800.0);

    node_points, elements, _, _ , fixities, load_cases = load_truss_json(problem_file)
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

    # * Before optimization, check ground mesh stability
    solver = FEASolver(Displacement, Direct, problem);
    solver()
    K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    @test isfinite(logdet(cholesky(K+G)))

    # API: A v = λ B v, B needs to be PD for numberical stability
    # K + λ G = 0 <=> K v = -λ G v
    # * 1/λ K + G = 0 <=> -G v = 1/λ K v
    # dense_eigvals, buckmodes = eigen(Array(-G),Array(K));
    # @show dense_smallest_pos_eigval = minimum(1 ./ dense_eigvals[findall(dense_eigvals.>0)]);

    sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
    smallest_pos_eigval = 1/sparse_eigvals[1]

    # @test smallest_pos_eigval == smallest_pos_eigval
    @test smallest_pos_eigval >= 1.0

    # * Run optimization, the optimized result should be unstable
    xmin = 0.0001 # minimum density
    x0 = fill(1.0, ncells) # initial design
    p = 4.0 # penalty
    V = 0.5 # maximum volume fraction

    comp = TopOpt.Compliance(problem, solver)
    function obj(x)
        # minimize compliance
        return comp(x)
    end
    function constr(x)
        # volume fraction constraint
        return sum(x) / length(x) - V
    end

    m = Model(obj)
    addvar!(m, zeros(length(x0)), ones(length(x0)))
    Nonconvex.add_ineq_constraint!(m, constr)

    options = MMAOptions(
        maxiter=1000, tol = Tolerance(kkt = 1e-4, f = 1e-4),
    )
    TopOpt.setpenalty!(solver, p)
    r = Nonconvex.optimize(
        m, MMA87(dualoptimizer = ConjugateGradient()),
        x0, options = options,
    );

    solver = FEASolver(Displacement, Direct, problem; xmin=xmin);
    solver.vars = r.minimizer;
    solver()
    K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    # check optimizing compliance without buckling constraint will lead to instable structures
    @test_throws PosDefException logdet(cholesky(K+G))

    # using Makie
    # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
    # fig = visualize(problem; topology=r.minimizer)
end