using Test
using TopOpt
using TopOpt.TopOptProblems: getE
using TopOpt.TrussTopOptProblems: load_truss_json, load_truss_geo
using Base.Iterators
using Makie
using CairoMakie
# using GLMakie

ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

@testset "Test parsing $file_format" for file_format in [".geo", ".json"]
    file_name = "tim_2d" * file_format
    problem_file = joinpath(ins_dir, file_name)
    mat = TrussFEAMaterial(1.0, 0.3)
    crossec = TrussFEACrossSec(800.0)
    if file_format == ".geo"
        node_points, elements, fixities, load_cases = load_truss_geo(problem_file)
        loads = load_cases[1]
    else
        node_points, elements, _, _, fixities, load_cases = load_truss_json(problem_file)
        loads = load_cases["0"]
    end
    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mat, crossec
    )

    solver = FEASolver(Direct, problem)
    solver()
    @test !all(@. isnan(solver.u))
end

@testset "Tim problem to solve" for (problem_dim, lc_ind) in product(["2d", "3d"], [0, 1])
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(ins_dir, file_name)

    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        problem_file
    )
    loads = load_cases[string(lc_ind)]

    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
    )

    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    @test getE(problem) == [m.E for m in mats]

    V = 0.3 # volume fraction
    xmin = 0.001 # minimum density
    rmin = 4.0 # density filter radius

    penalty = TopOpt.PowerPenalty(1.0) # 1
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    ## call solver to trigger assemble!
    solver()

    # * Compliance
    comp = TopOpt.Compliance(solver)
    obj = x -> comp(PseudoDensities(x))
    volfrac = TopOpt.Volume(solver)
    constr = x -> volfrac(PseudoDensities(x)) - V

    options = MMAOptions(; maxiter=3000, tol=Nonconvex.Tolerance(; kkt=0.001))
    x0 = fill(V, length(solver.vars))
    nelem = length(x0)

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    TopOpt.setpenalty!(solver, penalty.p)
    result = Nonconvex.optimize(m, MMA87(), x0; options=options)

    println("="^10)
    println(
        "tim-$(problem_dim) - LC $(lc_ind) - #elements $(ncells), #dof: $(ncells*ndim): opt iter $(result.iter)",
    )
    println("$(result.convstate)")

    solver()
    fig = visualize(
        problem;
        u=solver.u,
        topology=result.minimizer,
        vector_arrowsize=0.1,
        vector_linewidth=0.8,
        default_exagg_scale=ndim == 3 ? 1.0 : 0.01,
        exagg_range=ndim == 3 ? 10.0 : 0.1,
    )
    Makie.display(fig)
end # end testset

@testset "PointLoadCantileverTruss" for dim in [2, 3]
    nels = dim == 2 ? (10, 4) : (10, 4, 4)
    cell_size = Tuple(ones(Float32, dim))

    problem = PointLoadCantileverTruss(nels, cell_size; k_connect=1)

    V = 0.1 # volume fraction
    xmin = 0.001 # minimum density
    rmin = 4.0 # density filter radius

    penalty = TopOpt.PowerPenalty(1.0) # 1
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    ## call solver to trigger assemble!
    solver()

    # * Compliance
    comp = TopOpt.Compliance(solver)
    obj = x -> comp(PseudoDensities(x))
    volfrac = TopOpt.Volume(solver)
    constr = x -> volfrac(PseudoDensities(x)) - V

    options = MMAOptions(; maxiter=3000, tol=Nonconvex.Tolerance(; kkt=0.001))
    x0 = fill(V, length(solver.vars))
    nelem = length(x0)

    m = Model(obj)
    addvar!(m, zeros(nelem), ones(nelem))
    add_ineq_constraint!(m, constr)

    TopOpt.setpenalty!(solver, penalty.p)
    result = Nonconvex.optimize(m, MMA87(), x0; options=options)

    fig = visualize(problem; topology=result.minimizer)
    Makie.display(fig)
end # end testset
