"""
problem
2D Truss structure with 3 nodes and 2 elements

Input JSON file name "testfile2.json"
"""

# Draw a 3 member truss using / \ to represent the members
# Truss is a 2D truss structure with 3 nodes and 3 elements
#         2
#         /\
#    (1) /  \ (3)
#       /    \
#      1______3 
#         (2)
#     (pin)  (roller)

# Point coordinates
# 1 -> (0,0)
# 2 -> (5,5)
# 3 -> (10,0)

# Force is -100 unit in y direction at node 2

# The result forces/stress (every element area = 1) in the elements are 
# 1 -> -70.7106 [50sqrt(2) , compression]
# 2 -> +50.0000 [50 , tension]
# 3 -> -70.7106 [50sqrt(2) , compression]

using TopOpt
using Makie
using CairoMakie
# using GLMakie
using ColorSchemes

@testset "TrussStress" begin
    # Hand calculated result.
    result_stress = [-50 * sqrt(2.0), 50.0000, -50 * sqrt(2.0)]
    # Data input

    # Load the JSON file using load_truss_json
    # Geometric connection
    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        joinpath(@__DIR__, "testfile2_compact.json")
    )
    # Problem setup
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    # Get the load case 0
    loads = load_cases["0"]
    # Assemble the problem
    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
    )

    # println("This problem has ", nnodes, " nodes and ", ncells, " elements.")

    # Dummy Density vector so PseudoDensities works
    x = ones(ncells, 1)[:, 1]
    #set xmin for FEASolver
    xmin = 0.0001
    # Set the solver and solve
    solver = FEASolver(Direct, problem; xmin=xmin)
    # Get the stress
    ts = TrussStress(solver)
    σ = ts(PseudoDensities(x))

    @show σ
    # Check the result
    @assert abs(σ[1] - result_stress[1]) < 1e-12
    @assert abs(σ[2] - result_stress[2]) < 1e-12
    @assert abs(σ[3] - result_stress[3]) < 1e-12

    #visualization
    color_per_cell = abs.(σ .* x)
    fig1 = visualize(
        problem;
        u=fill(0.1, nnodes * ndim),
        topology=x,
        default_exagg_scale=0.0,
        default_element_linewidth_scale=5.0,
        default_load_scale=0.5,
        default_support_scale=0.1,
        cell_colors=color_per_cell,
        colormap=ColorSchemes.Spectral_10,
    )
    Makie.display(fig1)
end
