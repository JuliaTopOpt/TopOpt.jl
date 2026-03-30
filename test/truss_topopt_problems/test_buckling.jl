using Test
using LinearAlgebra
using StaticArrays
using Ferrite
using SparseArrays

using TopOpt
using TopOpt.TrussTopOptProblems.TrussTopOptProblems: compute_local_axes
using TopOpt.TopOptProblems: getE, getdh
using TopOpt.TrussTopOptProblems: getA
using Arpack

include("utils.jl")

@testset "buckling function with displacement argument" begin
    # Load a simple truss problem
    fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples")
    problem_file = joinpath(fea_ins_dir, "mgz_geom_stiff_ex9.1.json")
    
    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file)
    loads = load_cases["0"]
    
    problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs)
    solver = FEASolver(DirectSolver, problem)
    solver()
    
    # Test buckling with u argument (lines 89-93 in truss_topoptproblems.jl)
    u = solver.u
    Ke, Kg = buckling(problem, solver.globalinfo, solver.elementinfo; u=u)
    
    # Verify return types and dimensions
    # The exact type depends on the problem - just check they are valid matrices
    @test !isnothing(Ke)
    @test !isnothing(Kg)
    
    # Verify that the combined matrix is valid
    K = Ke + Kg
    @test size(K, 1) == size(K, 2)
    
    # Test buckling without u argument (covers lines 91-92: u === undef branch)
    Ke2, Kg2 = buckling(problem, solver.globalinfo, solver.elementinfo)
    @test !isnothing(Ke2)
    @test !isnothing(Kg2)
    @test size(Ke2) == size(Ke)
    @test size(Kg2) == size(Kg)
end

fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples");
gm_ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

# @testset "large displacement simulation MGZ-ex 9.1" begin
#     problem_file = joinpath(fea_ins_dir, "mgz_geom_stiff_ex9.1.json")

#     node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file);

#     P_val = 400 # kN
#     P = Dict(2 => SVector{2}(P_val * [0.05, -1.0]))
#     # P = load_cases["0"]

#     maxit = 100
#     utol = 1e-5
#     x0 = fill(1.0, length(elements)) # initial design

#     # * initialize displacements as zero
#     nnode = length(node_points)
#     u0 = zeros(nnode*2)

#     # * additional fields
#     u1 = deepcopy(u0)
#     rhs = deepcopy(u0)

#     step = 0
#     load_parameters = 0.0:0.05:0.5;
#     # tracking iteration history for plotting
#     tip_displacement = fill(0.0, length(load_parameters), 2);

#     # TODO Newton-Raphson Iterations
#     # https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
#     # https://kristofferc.github.io/JuAFEM.jl/dev/examples/hyperelasticity/
#     for load_parameter in load_parameters
#         # guess
#         u1[:] = u0[:]
#         P = Dict(2 => SVector{2}(load_parameter * P_val * [0.05, -1.0]))

#         println("Load: $load_parameter")
#         iter = 1;
#         while true
#             # internal force based on previous configuration
#             # assemble K based on the conf (orig + u1)
#             updated_nodes = Dict(nid => pt + u1[2*nid-1:2*nid] for (nid, pt) in node_points)
#             problem = TrussProblem(Val{:Linear}, updated_nodes, elements, P, fixities, mats, crosssecs);
#             solver = FEASolver(DirectSolver, problem)
#             # trigger assembly
#             solver()

#             assemble_k = TopOpt.AssembleK(problem)
#             element_k = ElementK(solver)
#             truss_element_kσ = TrussElementKσ(problem, solver)

#             # reaction force in global coordinate
#             Fr = solver.globalinfo.K * u1
#             # external force
#             F = solver.globalinfo.f
#             @. rhs = F - Fr;

#             # # ! check geometric stiffness matirx
#             # * u_e, x_e -> Ksigma_e
#             Kσs_1 = truss_element_kσ(u1, x0)
#             Kσs_2 = get_truss_Kσs(problem, u1, solver.elementinfo.cellvalues)

#             As = getA(problem)
#             Es = getE(problem)
#             for cell in CellIterator(getdh(problem))
#                 cellidx = cellid(cell)
#                 global_dofs = celldofs(cell)

#                 coords = getcoordinates(cell)
#                 L = norm(coords[1] - coords[2])
#                 A = As[cellidx]
#                 E = Es[cellidx]

#                 R2 = compute_local_axes(cell.coords[1], cell.coords[2])
#                 R4 = zeros(4,4)
#                 R4[1:2,1:2] = R2
#                 R4[3:4,3:4] = R2

#                 γ = vcat(-R2[:,1], R2[:,1])
#                 u_cell = @view u1[global_dofs]
#                 q_cell = E*A*(γ'*u_cell/L)

#                 # ? why MGZ formula different?
#                 # Kg_m = R4'*(q_cell[2]/L)*[1 0 -1 0; 
#                 #                            0 1 0 -1; 
#                 #                            -1 0 1 0; 
#                 #                            0 -1 0 1]*R4
#                 Kg_m = R4*(q_cell/L)* [0 0 0 0; 
#                                         0 1 0 -1; 
#                                         0 0 0 0; 
#                                         0 -1 0 1]*R4'
#                 @test Kg_m ≈ Kσs_1[cellidx]
#                 @test Kg_m ≈ Kσs_2[cellidx] 
#             end

#             # solve for the new deformation
#             Ke, Kg = buckling(problem, solver.globalinfo, solver.elementinfo; u=u1)
#             K = Ke + Kg
#             dchi = K\rhs
#             u1[:] += dchi[:]

#             print("$iter: ||du||=$(maximum(abs.(dchi[:])))\n")
#             if maximum(abs.(dchi[:])) < utol # convergence check
#                 break;
#             end
#             if (iter > maxit)# bailout for failed convergence
#                 error("Possible failed convergence");
#             end
#             iter += 1;
#         end
#         u0[:] = u1[:];       # update the displacement

#         step = step + 1
#         # save displacement record for plotting
#         tip_displacement[step, :] .= u1[2*2-1:2*2]
#     end

#     # using Makie
#     # fig, ax, p = scatter(tip_displacement[:,1], load_parameters.*P_val,
#     #     axis = (xlabel = "u_b (m)", ylabel = "P (kN)"))
#     # lines!(tip_displacement[:,1], load_parameters.*P_val, color = :blue, linewidth = 3)
#     # fig

#     # fig = visualize(problem; u=u1)
# end # end test set
