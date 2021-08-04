using Test
using LinearAlgebra
using Ferrite

using TopOpt
using TopOpt.TrussTopOptProblems: buckling
# using IterativeSolvers
# using Arpack
# using Crayons.Box

# problem_json = ["buckling_2d_nodal_instab.json", "buckling_2d_global_instab.json", "buckling_2d_debug.json"]
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
    @time r = Nonconvex.optimize(
        m, MMA87(dualoptimizer = ConjugateGradient()),
        x0, options = options,
    );

    solver = FEASolver(Displacement, Direct, problem; xmin=xmin);
    solver.vars = r.minimizer;
    solver()
    K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    # check optimizing compliance without buckling constraint will lead to instable structures
    @test_throws PosDefException logdet(cholesky(K+G))
end

# # @testset "Buckling problem solve - $(problem_json[i])" for i in 1:length(problem_json)
#     i = 2
#     file_name = problem_json[i]
#     problem_file = joinpath(ins_dir, file_name)

#     node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file);
#     ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
#     loads = load_cases["0"]

#     problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs);

#     solver = FEASolver(Displacement, Direct, problem)
#     # call solver to trigger assemble!
#     solver()
#     @show solver.u
#     scene, layout = draw_truss_problem(problem, solver.u;
#         default_load_scale=0.2, default_support_scale=0.2, vector_arrowsize=0.03)

#     try
#         global K, Kσ = buckling(problem, solver.globalinfo, solver.elementinfo)
#     catch err
#         println(RED_BG("ERROR: "))
#         println(err)
#         if isa(err, SingularException)
#             println(RED_FG("Linear elasticity solve failed due to mechanism."))
#             K_l = solver.globalinfo.K
#             # TODO: use sparse eigen
#             # λ, ϕ = eigs(K.data, nev = 2, which=:SM);
#             # @show λ
#             # @show ϕ

#             # ? is this ordered by eigen values' magnitude?
#             local F = eigen(Array(K_l))
#             @show F
#             @assert abs(F.values[1] - 0.0) < eps()

#             # * draw eigen mode
#             scene, layout = draw_truss_problem(problem; u=F.vectors[:,1])
#         end
#         throw(err)
#     end
#     # the linear elastic model must be solved successfully to proceed on buckling calculation
#     println(GREEN_FG("Linear elasticity solved."))

#     ch = problem.ch
#     C = zeros(size(K, 1), length(ch.prescribed_dofs))
#     setindex!.(Ref(C), 1, ch.prescribed_dofs, 1:length(ch.prescribed_dofs))
#     # C = hcat(C, solver.u)

#     # Find the maximum eigenvalue of system Kσ x = 1/λ K x
#     # The reason we do it this way is because K is guaranteed to be positive definite while Kσ is not and the LOBPCG algorithm to find generalised eigenvalues requires the matrix on the RHS to be positive definite.
#     # https://julialinearalgebra.github.io/IterativeSolvers.jl/stable/eigenproblems/lobpcg/
#     c = norm(K)
#     # r = lobpcg(Kσ.data, K.data, true, 1; C=C)
#     r = lobpcg(Kσ.data, K.data, true, 1)
#     @show r

#     # # Minimum eigenvalue of the system K x + λ Kσ x = 0
#     @show λ = -1/r.λ[1]
#     @show v = r.X

#     scene, layout = draw_truss_problem(problem; u=v, default_exagg=1.0, exagg_range=10.0,
#         default_load_scale=0.2, default_support_scale=0.2, default_arrow_size=0.03)

#     # F = eigen(Array(Kσ), Array(K))
#     # F = eigen(Array(K), Array(Kσ))
#     # @show - 1 ./ F.values

#     if 0 < λ && λ < 1
#         # 0 < λ_min < 1 means that the equilibrium equation with load 
#         # λ*f is not solvable for some 0 < λ < 1 and the structure under load
#         # f is not stable

#         # s = 1.0
#         Ks = K .+ λ .* Kσ

#         eigen_vals, eigen_vecs = eigs(Ks, nev = 2, which=:SM);
#         @show eigen_vals
#         @show eigen_vecs
#         local F = eigen(Array(Ks))
#         @show F

#         # cholKs = cholesky(Symmetric(Ks))
#         # u_b = cholKs \ solver.globalinfo.f
#         # println(GREEN_FG("Buckling problem solved."))

#         ## TODO plot analysis result with
#         # scene, layout = draw_truss_problem(problem; u=u_b)
#         # println("End")
#     end

#     # TODO eigenmode to show global instability mode
#     # Find a test case for verifying the geometric stiffness matrix
#     # ? what does MMA need? Inner opt uses MMA or SIMP?
#     #  - where to plug in the gradient of the barrier function?
#         # function (v::Volume{T})(x, grad = nothing) where {T}
#     #  - how to compute ∂K/∂x_e and ∂f/∂x_e?
#             # d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
#             # if PENALTY_BEFORE_INTERPOLATION
#             #     p = density(penalty(d), xmin)
#             # else
#             #     p = penalty(density(d, xmin))
#             # end
#             # grad[varind[i]] = -p.partials[1] * cell_comp[i] # Ke
#     #  adjoint method: ∂u/∂x_e = K^{-1} * (∂f/∂x_e - ∂K/∂x_e * u)
#     # ∂Kσ/∂x_e for all e

#     # TODO finite difference to verify the gradient
#     # TODO verify the gradient for some analytical problems
#     # TODO "manual" interior point loop, adjusting the c value every iter

# # end # end test set

# =#