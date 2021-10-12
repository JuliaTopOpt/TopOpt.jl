using Test
using LinearAlgebra
using Ferrite

using TopOpt
using TopOpt: _apply!
using TopOpt.TrussTopOptProblems: buckling
using Arpack

fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples");
gm_ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

@testset "Tim buckling log-barrier $problem_dim" for problem_dim in ["2d"] # , "3d"
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(gm_ins_dir, file_name)

    mats = TrussFEAMaterial(1.0, 0.3);
    crossecs = TrussFEACrossSec(800.0);

    node_points, elements, _, _ , fixities, load_cases = load_truss_json(problem_file)
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

    xmin = 0.0001 # minimum density
    p = 4.0 # penalty
    V = 0.5 # maximum volume fraction
    x0 = fill(1.0, ncells) # initial design

    solver = FEASolver(Direct, problem);

    # * Before optimization, check initial design stability
    solver.vars = x0
    solver()
    K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    @test isfinite(logdet(cholesky(K+G)))
    sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
    smallest_pos_eigval = 1/sparse_eigvals[1]
    @test smallest_pos_eigval >= 1.0

    Nonconvex.show_residuals[] = true

    comp = TopOpt.Compliance(problem, solver)
    for c in [0.1] # 10:-0.1:0.1
        function obj(x)
            solver.vars = x
            # trigger assembly
            solver()
            K, G = buckling(problem, solver.globalinfo, solver.elementinfo);
            # minimize compliance
            return comp(x) - c*logdet(cholesky(Array(K+G)))
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
    end

    # check result stability
    solver = FEASolver(Direct, problem; xmin=xmin);
    solver.vars = r.minimizer;
    solver()
    K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    @test isfinite(logdet(cholesky(K+G)))
    sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
    smallest_pos_eigval = 1/sparse_eigvals[1]
    @test smallest_pos_eigval >= 1.0

    # using Makie
    # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
    # fig = visualize(problem; topology=r.minimizer)
end

@testset "Tim buckling SDP constraint $problem_dim" for problem_dim in ["2d"] # , "3d"
    file_name = "tim_$(problem_dim).json"
    problem_file = joinpath(gm_ins_dir, file_name)

    mats = TrussFEAMaterial(1.0, 0.3);
    crossecs = TrussFEACrossSec(800.0);

    node_points, elements, _, _ , fixities, load_cases = load_truss_json(problem_file)
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

    xmin = 0.0001 # minimum density
    p = 4.0 # penalty
    V = 0.5 # maximum volume fraction
    x0 = fill(1.0, ncells) # initial design

    solver = FEASolver(Direct, problem);
    ch = problem.ch
    dh = problem.ch.dh

    # # * Before optimization, check initial design stability
    # solver.vars = x0
    # solver()
    # K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    # @test isfinite(logdet(cholesky(K+G)))
    # sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
    # smallest_pos_eigval = 1/sparse_eigvals[1]
    # @test smallest_pos_eigval >= 1.0

    Nonconvex.show_residuals[] = true

    comp = TopOpt.Compliance(problem, solver)
    dp = TopOpt.Displacement(solver)
    assemble_k = TopOpt.AssembleK(problem)
    cellvalues = solver.elementinfo.cellvalues
    ndof_pc = ndofs_per_cell(dh)
    element_K_fns = []
    element_Kσ_fns = []
    # constructing two functions for each element
    for cellidx = 1:ncells
        # element K (linear) constructor function: x_e -> K_e
        push!(element_K_fns, ElementK(solver, cellidx))
        # element Kσ constructor function: x_e -> Kσ_e
        push!(element_Kσ_fns, TrussElementKσ(problem, cellidx, cellvalues))
    end
    global_dofs = copy(element_Kσ_fns[1].global_dofs)

    # element stiffness matrices
    Ke0s = solver.elementinfo.Kes

    # * comliance minimization objective
    obj = comp
    c = 1.0 # buckling load multiplier

    function buckling_matrix_constr(x)
        # * solve for the displacement
        # n_dof of the entire system
        u = dp(x)

        # * x -> Kes, construct all the element stiffness matrices
        # a list of small matrices for each element (cell)
        Kes = []
        for e in 1:length(x)
            K_e_vec = element_K_fns[e](x[e])
            push!(Kes, reshape(Ksig_e_vec, (ndof_pc, ndof_pc)))
        end

        # dp(K)
        # u = K(x) \ f

        # * Kes -> K (global linear stiffness matrix)
        K = assemble_k(Kes)
        # * apply boundary condition
        apply!(K, ch)

        # * u_e, x_e -> Ksigma_e
        Kσs = []
        for e in 1:length(x)
            celldofs!(global_dofs, dh, e)
            Ksig_e_vec = element_Kσ_fns[e](u[global_dofs], x[e])
            push!(Kσs, reshape(Ksig_e_vec, (ndof_pc, ndof_pc)))
        end
        # * Kσs -> Kσ
        Kσ = assemble_k(Kσs)
        # * apply boundary condition
        apply_zero!(Kσ, ch)

        return Array(K + c*Kσ)
    end

    # Array(K + c*Kσ) >= 0, PSD

    function vol_constr(x)
        # volume fraction constraint
        return sum(x) / length(x) - V
    end

    m = Model(obj)
    addvar!(m, zeros(length(x0)), ones(length(x0)))
    Nonconvex.add_ineq_constraint!(m, vol_constr)
    Nonconvex.add_sd_constraint!(m, buckling_matrix_constr)

    # TODO use SDPAlg
    options = MMAOptions(
        maxiter=1000, tol = Tolerance(kkt = 1e-4, f = 1e-4),
    )
    TopOpt.setpenalty!(solver, p)
    r = Nonconvex.optimize(
        m, MMA87(dualoptimizer = ConjugateGradient()),
        x0, options = options,
    );

    # # * check result stability
    # solver = FEASolver(Direct, problem; xmin=xmin);
    # solver.vars = r.minimizer;
    # solver()
    # K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
    # @test isfinite(logdet(cholesky(K+G)))
    # sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
    # smallest_pos_eigval = 1/sparse_eigvals[1]
    # @test smallest_pos_eigval >= 1.0

    # using Makie
    # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
    # fig = visualize(problem; topology=r.minimizer)
end

#  - where to plug in the gradient of the barrier function?
    # function (v::Volume{T})(x, grad = nothing) where {T}
#  - how to compute ∂K/∂x_e and ∂f/∂x_e?
        # d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
        # if PENALTY_BEFORE_INTERPOLATION
        #     p = density(penalty(d), xmin)
        # else
        #     p = penalty(density(d, xmin))
        # end
        # grad[varind[i]] = -p.partials[1] * cell_comp[i] # Ke
#  adjoint method: ∂u/∂x_e = K^{-1} * (∂f/∂x_e - ∂K/∂x_e * u)
# ∂Kσ/∂x_e for all e

# TODO finite difference to verify the gradient
# TODO verify the gradient for some analytical problems
# TODO "manual" interior point loop, adjusting the c value every iter