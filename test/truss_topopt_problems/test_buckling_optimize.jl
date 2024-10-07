using Test
using LinearAlgebra
using Ferrite
using NonconvexIpopt

using TopOpt
using Arpack

using Makie
using CairoMakie
# using GLMakie

fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples");
gm_ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

# @testset "Tim buckling log-barrier $problem_dim" for problem_dim in ["2d"] # , "3d"
#     file_name = "tim_$(problem_dim).json"
#     problem_file = joinpath(gm_ins_dir, file_name)

#     mats = TrussFEAMaterial(1.0, 0.3);
#     crossecs = TrussFEACrossSec(800.0);

#     node_points, elements, _, _ , fixities, load_cases = load_truss_json(problem_file)
#     ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
#     loads = load_cases["0"]

#     problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

#     xmin = 0.0001 # minimum density
#     p = 4.0 # penalty
#     V = 0.5 # maximum volume fraction
#     x0 = fill(1.0, ncells) # initial design

#     solver = FEASolver(Direct, problem);

#     # # * Before optimization, check initial design stability
#     # solver.vars = x0
#     # solver()
#     # K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
#     # @test isfinite(logdet(cholesky(K+G)))
#     # sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
#     # smallest_pos_eigval = 1/sparse_eigvals[1]
#     # @test smallest_pos_eigval >= 1.0

#     comp = TopOpt.Compliance(solver)
#     # TODO "manual" interior point loop, adjusting the c value every iter
#     for c in [0.1] # 10:-0.1:0.1
#         function obj(x)
#             solver.vars = x
#             # trigger assembly
#             solver()
#             K, G = buckling(problem, solver.globalinfo, solver.elementinfo);
#             # minimize compliance
#             return comp(x) - c*logdet(cholesky(Array(K+G)))
#         end
#         function constr(x)
#             # volume fraction constraint
#             return sum(x) / length(x) - V
#         end

#         m = Model(obj)
#         addvar!(m, zeros(length(x0)), ones(length(x0)))
#         Nonconvex.add_ineq_constraint!(m, constr)

#         options = MMAOptions(
#             maxiter=1000, tol = Tolerance(kkt = 1e-4, f = 1e-4),
#         )
#         TopOpt.setpenalty!(solver, p)
#         r = Nonconvex.optimize(
#             m, MMA87(dualoptimizer = ConjugateGradient()),
#             x0, options = options,
#         );
#     end

#     # check result stability
#     solver = FEASolver(Direct, problem; xmin=xmin);
#     solver.vars = r.minimizer;
#     solver()
#     K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
#     @test isfinite(logdet(cholesky(K+G)))
#     sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
#     smallest_pos_eigval = 1/sparse_eigvals[1]
#     @test smallest_pos_eigval >= 1.0

#     # fig = visualize(problem; topology=r.minimizer)
# end

@testset "Tim buckling SDP constraint $problem_dim" for problem_dim in ["2d"] # , "3d"
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
    p = 1.0 # penalty
    V = 0.5 # maximum volume fraction
    x0 = fill(1.0, ncells) # initial design

    solver = FEASolver(Direct, problem)
    ch = problem.ch
    dh = problem.ch.dh

    comp = TopOpt.Compliance(solver)
    dp = TopOpt.Displacement(solver)
    assemble_k = TopOpt.AssembleK(problem)
    element_k = ElementK(solver)
    truss_element_kσ = TrussElementKσ(problem, solver)

    # * comliance minimization objective
    obj = x -> comp(PseudoDensities(x))
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

    function vol_constr(x)
        # volume fraction constraint
        return sum(x) / length(x) - V
    end

    # * Before optimization, check initial design stability
    @test isfinite(logdet(cholesky(buckling_matrix_constr(x0))))
    @test vol_constr(x0) == 0.5

    m = Model(obj)
    addvar!(m, zeros(length(x0)), ones(length(x0)))
    Nonconvex.add_ineq_constraint!(m, vol_constr)
    alg = IpoptAlg()
    options = IpoptOptions(; max_iter=200)
    r1 = Nonconvex.optimize(m, alg, x0; options=options)
    @test vol_constr(r1.minimizer) < 1e-7

    Nonconvex.add_sd_constraint!(m, buckling_matrix_constr)
    alg = SDPBarrierAlg(; sub_alg=IpoptAlg())
    options = SDPBarrierOptions(; sub_options=IpoptOptions(; max_iter=200), keep_all=true)
    r2 = Nonconvex.optimize(m, alg, x0; options=options)
    @test vol_constr(r2.minimizer) < 1e-7

    # * check result stability
    S0 = buckling_matrix_constr(x0)
    S1 = buckling_matrix_constr(r1.minimizer)
    S2 = buckling_matrix_constr(r2.minimizer)
    ev0 = eigen(S0).values
    ev1 = eigen(S1).values
    ev2 = eigen(S2).values

    @test isfinite(logdet(cholesky(S0)))
    @test minimum(ev0) ≈ 0.34 rtol = 0.01
    @test maximum(ev0) ≈ 4204 rtol = 0.01

    @test_throws PosDefException cholesky(S1)

    @test isfinite(logdet(cholesky(S2)))
    @test 0 < minimum(ev2) < 0.02
    @test maximum(ev2) ≈ 3250 rtol = 0.001

    fig = visualize(problem; topology=x0)
    Makie.display(fig)

    fig = visualize(problem; topology=r2.minimizer)
    Makie.display(fig)
end
