using Test
using LinearAlgebra
using LinearAlgebra: norm
using Ferrite
using StaticArrays

using TopOpt
using TopOpt.TrussTopOptProblems.TrussTopOptProblems
using TopOpt.TrussTopOptProblems.TrussIO: load_truss_geo, load_truss_json
using TopOpt.TopOptProblems: boundingbox, nnodespercell, getgeomorder, getmetadata, getdh, getE, getdim
using TopOpt.TrussTopOptProblems: getA, default_quad_order
using TopOpt.TrussTopOptProblems.TrussTopOptProblems: buckling, get_truss_Kσs, compute_local_axes

ins_dir = joinpath(@__DIR__, "instances");
problem_json = ["ground_meshes\\tim_2d.json"]

@testset "large displacement simulation MGZ-ex 9.1" begin
    file_name = problem_json[i]
    problem_file = joinpath(ins_dir, "fea_examples\\mgz_geom_stiff_ex9.1.json")

    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file);

    # P_val = 400 # kN
    # P = Dict(2 => SVector{2}(P_val * [0.05, -1.0]))
    P = load_cases["0"]

    maxit = 400
    utol = 1e-5

    # * initialize displacements as zero
    nnode = length(node_points)
    u0 = zeros(nnode*2)

    # * additional fields
    u1 = deepcopy(u0)
    rhs = deepcopy(u0)

    step = 0
    load_parameters = 0.0:0.05:0.5;
    # tracking iteration history for plotting
    tip_displacement = fill(0.0, length(load_parameters), 2);

    # TODO Newton-Raphson Iterations
    # https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
    # https://kristofferc.github.io/JuAFEM.jl/dev/examples/hyperelasticity/
    for load_parameter in load_parameters
        # guess
        u1[:] = u0[:]
        P = Dict(2 => SVector{2}(load_parameter * P_val * [0.05, -1.0]))

        println("Load: $load_parameter")
        iter = 1;
        while true
            # internal force based on previous configuration
            # assemble K based on the conf (orig + u1)
            updated_nodes = Dict(nid => pt + u1[2*nid-1:2*nid] for (nid, pt) in node_points)
            problem = TrussProblem(Val{:Linear}, updated_nodes, elements, P, fixities, mats, crosssecs);
            solver = FEASolver(Direct, problem)
            # trigger assembly
            solver()

            # reaction force in global coordinate
            Fr = solver.globalinfo.K * u1
            # external force
            F = solver.globalinfo.f
            @. rhs = F - Fr;

            # # ! check geometric stiffness matirx
            Kσs = get_truss_Kσs(problem, u1, solver.elementinfo.cellvalues)
            As = getA(problem)
            Es = getE(problem)
            for cell in CellIterator(getdh(problem))
                cellidx = cellid(cell)
                global_dofs = celldofs(cell)

                coords = getcoordinates(cell)
                L = norm(coords[1] - coords[2])
                A = As[cellidx]
                E = Es[cellidx]

                R2 = compute_local_axes(cell.coords[1], cell.coords[2])
                R4 = zeros(4,4)
                R4[1:2,1:2] = R2
                R4[3:4,3:4] = R2

                γ = vcat(-R2[:,1], R2[:,1])
                u_cell = @view u1[global_dofs]
                q_cell = E*A*(γ'*u_cell/L)

                # ? why MGZ formula different?
                # Kg_m = R4'*(q_cell[2]/L)*[1 0 -1 0; 
                #                            0 1 0 -1; 
                #                            -1 0 1 0; 
                #                            0 -1 0 1]*R4
                Kg_m = R4*(q_cell/L)* [0 0 0 0; 
                                        0 1 0 -1; 
                                        0 0 0 0; 
                                        0 -1 0 1]*R4'
                Kg = Kσs[cellidx]
                @test Kg_m ≈ Kg
            end

            # solve for the new deformation
            Ke, Kg = buckling(problem, solver.globalinfo, solver.elementinfo; u=u1)
            K = Ke + Kg
            dchi = K\rhs
            u1[:] += dchi[:]

            print("$iter: ||du||=$(maximum(abs.(dchi[:])))\n")
            if maximum(abs.(dchi[:])) < utol # convergence check
                break;
            end
            if (iter > maxit)# bailout for failed convergence
                error("Possible failed convergence");
            end
            iter += 1;
        end
        u0[:] = u1[:];       # update the displacement
        
        step = step + 1
        # save displacement record for plotting
        tip_displacement[step, :] .= u1[2*2-1:2*2]
    end

    # using Makie
    # fig, ax, p = scatter(tip_displacement[:,1], load_parameters.*P_val,
    #     axis = (xlabel = "u_b (m)", ylabel = "P (kN)"))
    # lines!(tip_displacement[:,1], load_parameters.*P_val, color = :blue, linewidth = 3)
    # fig

    # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
    # fig = visualize(problem, u1)
end # end test set