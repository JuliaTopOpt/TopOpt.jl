using Test
using TopOpt
using TopOpt.TopOptProblems: getE
using TopOpt.TrussTopOptProblems: load_truss_json, load_truss_geo
using Base.Iterators
# if get(ENV, "CI", nothing) != "true"
#     import Makie
#     @eval using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
# end

ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

@testset "Test parsing $file_format" for file_format in [".geo", ".json"]
    file_name = "tim_2d"*file_format
    problem_file = joinpath(ins_dir, file_name)
    mat = TrussFEAMaterial(1.0, 0.3);
    crossec = TrussFEACrossSec(800.0);
    if file_format == ".geo"
        node_points, elements, fixities, load_cases = load_truss_geo(problem_file)
        loads = load_cases[1]
    else
        node_points, elements, _, _, fixities, load_cases = load_truss_json(problem_file);
        loads = load_cases["0"]
    end
    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mat, crossec);

    solver = FEASolver(Displacement, Direct, problem);
    solver()
    @test !all(@. isnan(solver.u))
end

@testset "Tim problem to solve" for (problem_dim, lc_ind) in product(["2d", "3d"], [0, 1])
    # problem_dim = "2d"
    # file_name = "tim_2d.json"
    # lc_ind = 1
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(ins_dir, file_name)

    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file);
    loads = load_cases[string(lc_ind)]

    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs);

    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    @test getE(problem) == [m.E for m in mats]

    V = 0.3 # volume fraction
    xmin = 0.001 # minimum density
    rmin = 4.0; # density filter radius

    penalty = TopOpt.PowerPenalty(1.0) # 1
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin,
        penalty = penalty);
    ## call solver to trigger assemble!
    solver()

    # * Compliance
    comp = TopOpt.Compliance(problem, solver)
    obj = comp
    volfrac = TopOpt.Volume(problem, solver)
    constr = x -> volfrac(x) - V

    mma_options = options = Nonconvex.MMAOptions(
        maxiter = 3000, tol = Nonconvex.Tolerance(kkt = 0.001),
    )
    convcriteria = Nonconvex.KKTCriteria()
    x0 = fill(V, length(solver.vars))
    nelem = length(x0)

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    # TODO try Mixed-Integer Algorithm
    # alg = JuniperIpoptAlg()
    # options = Nonconvex.JuniperIpoptOptions()
    # r = Nonconvex.optimize(m, alg, x0, options = options, integers = trues(nelem))
    # r.minimum
    # r.minimizer

    optimizer = Optimizer(
        obj, constr, x0, Nonconvex.MMA87(),
        options = mma_options, convcriteria = convcriteria,
    )
    simp = SIMP(optimizer, solver, penalty.p)

    # ? 1.0 might induce an infeasible solution, which gives the optimizer a hard time to escape 
    # from infeasible regions and return a result
    result = simp(x0);

    println("="^10)
    println("tim-$(problem_dim) - LC $(lc_ind) - #elements $(ncells), #dof: $(ncells*ndim): opt iter $(simp.optimizer.workspace.iter)")
    println("$(result.convstate)")

    solver()

    # if get(ENV, "CI", nothing) != "true"
    #     fig = visualize(
    #         problem, solver.u; topology = result.topology, vector_arrowsize = 0.1,
    #         vector_linewidth=0.8, default_exagg_scale=ndim == 3 ? 1.0 : 0.01,
    #         exagg_range = ndim == 3 ? 10.0 : 0.1,
    #     )
    #     Makie.display(fig)
    # end
end # end testset