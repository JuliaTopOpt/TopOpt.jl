using Test
using TopOpt
using TopOpt.TopOptProblems: getE
using Base.Iterators
using TopOpt.TrussTopOptProblems.TrussVisualization: visualize

ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

@testset "Tim problem to solve" for (problem_dim, lc_ind) in product(["2d", "3d"], [0, 1])
    # problem_dim = "2d"
    # file_name = "tim_2d.json"
    # lc_ind = 1
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(ins_dir, file_name)

    node_points, elements, mats, crosssecs, fixities, load_cases = parse_truss_json(problem_file);
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

    # ##############################
    # #! buckling
    # import TrussTopOpt.TrussTopOptProblems: default_quad_order
    # einfo = ElementFEAInfo(problem, TrussTopOpt.TrussTopOptProblems.default_quad_order(problem), Val{:Static})
    # ginfo = GlobalFEAInfo(problem)

    # using TrussTopOpt.TrussTopOptProblems: buckling, get_Kσs
    # buckling(problem, solver.globalinfo, solver.elementinfo)

    # ##############################

    # TODO TopOpt.LogBarrier
    # TODO linear_elasticity, du/dx
    # * Compliance
    obj = Objective(TopOpt.Compliance(problem, solver, filterT = nothing,
        rmin = rmin, tracing = true, logarithm = false));

    constr = Constraint(TopOpt.Volume(problem, solver, filterT = nothing, rmin = rmin), V);

    mma_options = options = MMA.Options(maxiter = 3000,
        tol = MMA.Tolerances(kkttol = 0.001))
    convcriteria = MMA.KKTCriteria()
    optimizer = MMAOptimizer(obj, constr, MMA.MMA87(),
        ConjugateGradient(), options = mma_options,
        convcriteria = convcriteria);

    simp = SIMP(optimizer, penalty.p);

    # ? 1.0 might induce an infeasible solution, which gives the optimizer a hard time to escape 
    # from infeasible regions and return a result
    x0 = fill(V, length(solver.vars))
    result = simp(x0);

    println("="^10)
    println("tim-$(problem_dim) - LC $(lc_ind) - #elements $(ncells), #dof: $(ncells*ndim): opt iter $(result.fevals)")
    println("$(result.convstate)")

    solver()

    fig = visualize(problem, solver.u; crosssecs=result.topology, vector_arrowsize=0.1, vector_linewidth=0.8, 
        default_exagg_scale=ndim == 3 ? 1.0 : 0.01, exagg_range=ndim == 3 ? 10.0 : 0.1)

    # import Makie
    # Makie.display(fig)

end # end testset