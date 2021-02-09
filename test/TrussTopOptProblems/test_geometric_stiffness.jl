using Test
using LinearAlgebra
using LinearAlgebra: norm
using JuAFEM
using StaticArrays

using TopOpt
using TrussTopOpt.TrussTopOptProblems
using TrussTopOpt.TrussTopOptProblems: buckling
using Crayons.Box
using GLMakie

problem_json = ["mgz_geom_stiff_ex9.1.json", "buckling_2d_debug.json"]
ins_dir = joinpath(@__DIR__, "instances", "fea_examples");

# @testset "Buckling problem solve - $(problem_json[i])" for i in 1:length(problem_json)
    i = 1
    file_name = problem_json[i]
    problem_file = joinpath(ins_dir, file_name)

    node_points, elements, mats, crosssecs, fixities, _ = parse_truss_json(problem_file);

    P_val = 400 # kN
    P = Dict(2 => SVector{2}(P_val * [0.05, -1.0]))
    nnode = length(node_points)

    # Newton iteration setup
    max_newton_itr = 50
    NEWTON_TOL = 1e-8

    function dict2vec(dict, veclen)
        v = zeros(veclen)
        for (nid, val) in dict
            v[2*nid-1:2*nid] = val
        end
        return v
    end

    # Δu = Dict(nid => SVector{2,Float64}(0,0) for nid in keys(node_points))

    updated_nodes = copy(node_points)
    problem = TrussProblem(Val{:Linear}, node_points, elements, P, fixities, mats, crosssecs);
    solver = FEASolver(Displacement, Direct, problem)
    solver()
    K, Kσ = buckling(problem, solver.globalinfo, solver.elementinfo; u=zeros(2*nnode))
    Kt = K + Kσ
    P_vec = solver.globalinfo.f
    # * initialize displacements as zero
    ut = zeros(nnode*2)

    # tracking iteration history for plotting
    ub_t = []
    p_t = []

    newton_itr = -1

    # Newton-Raphson Iterations
    # https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
    # https://kristofferc.github.io/JuAFEM.jl/dev/examples/hyperelasticity/
    while true
        global newton_itr, Kt, P_vec, ut, updated_nodes
        newton_itr += 1
        reaction = Kt * ut
        @show res_norm = norm(reaction - P_vec)
        push!(ub_t, ut[2*2-1])
        push!(p_t,-reaction[2*2])

        if res_norm < NEWTON_TOL
            break
        elseif newton_itr > max_newton_itr
            break
            # error("Reached maximum Newton iterations, aborting")
        end

        # construct new stiffness matrix
        ut = Kt \ P_vec
        updated_nodes = Dict(nid => pt + ut[2*nid-1:2*nid] for (nid, pt) in updated_nodes)
        problem = TrussProblem(Val{:Linear}, updated_nodes, elements, P, fixities, mats, crosssecs);
        solver = FEASolver(Displacement, Direct, problem)
        solver()
        K, Kσ = buckling(problem, solver.globalinfo, solver.elementinfo; u=ut)
        Kt = K + Kσ
    end
    scatter(ub_t, p_t)
    lines!(ub_t, p_t, color = :blue, linewidth = 3)
    current_figure()

    # scene, layout = draw_truss_problem(problem; u=v, default_exagg=1.0, exagg_range=10.0,
    #     default_load_scale=0.2, default_support_scale=0.2, default_arrow_size=0.03)

# end # end test set

# =#