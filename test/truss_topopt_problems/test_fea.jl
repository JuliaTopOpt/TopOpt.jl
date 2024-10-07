using Test
using LinearAlgebra
using Ferrite
using Ferrite: cellid, getcoordinates, CellIterator

using TopOpt
using TopOpt.TopOptProblems:
    boundingbox, nnodespercell, getgeomorder, getmetadata, getdh, getE, getdim
using TopOpt.TrussTopOptProblems: getA, default_quad_order, compute_local_axes
using Makie
using CairoMakie
# using GLMakie

include("utils.jl")

problem_json = ["mgz_truss1.json", "mgz_truss2.json", "mgz_truss3.json"]
u_solutions = [
    1e-3 * vcat([2.41, 0.72], zeros(2 * 2)),
    1e-3 * vcat([0.1783, 2.7222, -0.4863], zeros(4 * 3)),
    1e-3 * vcat([0.871, 1.244], [0, 0], [-0.193, 0]),
]
ins_dir = joinpath(@__DIR__, "instances", "fea_examples");

@testset "Truss problem solve - $(problem_json[i])" for i in 1:length(problem_json)
    # i = 3
    file_name = problem_json[i]
    problem_file = joinpath(ins_dir, file_name)

    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(
        problem_file
    )
    ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
    loads = load_cases["0"]

    problem = TrussProblem(
        Val{:Linear}, node_points, elements, loads, fixities, mats, crosssecs
    )

    @test getdim(problem) == ndim
    @test Ferrite.getncells(problem) == ncells
    @test getE(problem) == [m.E for m in mats]
    @test problem.black == problem.white == falses(ncells)
    @test problem.force == loads
    @test problem.varind == 1:ncells
    grid = problem.ch.dh.grid
    @test length(grid.cells) == ncells

    @test getgeomorder(problem) == 1
    @test nnodespercell(problem) == 2

    quad_order = default_quad_order(problem)
    elementinfo = ElementFEAInfo(problem, quad_order, Val{:Static})
    As = getA(problem)
    Es = getE(problem)
    for cell in CellIterator(getdh(problem))
        cellidx = cellid(cell)
        coords = getcoordinates(cell)
        L = norm(coords[1] - coords[2])
        A = As[cellidx]
        E = Es[cellidx]
        @test elementinfo.cellvolumes[cellidx] ≈ L * A

        Γ = zeros(2, ndim * 2)
        R = compute_local_axes(coords[1], coords[2])
        Γ[1, 1:ndim] = R[:, 1]
        Γ[2, (ndim + 1):(2 * ndim)] = R[:, 1]

        Ke_m = (A * E / L) * Γ' * [1 -1; -1 1] * Γ
        Ke = elementinfo.Kes[cellidx]
        @test Ke_m ≈ Ke
    end

    solver = FEASolver(Direct, problem)
    solver()

    fig = visualize(problem; u=solver.u)
    Makie.display(fig)

    # we use kN for force and m for length
    # thus, pressure/modulus is in kN/m
    # the textbook uses a quite rough rounding scheme...
    # 0.3 mm error
    to_K_full = solver.globalinfo.K.data
    @assert norm(solver.u - u_solutions[i]) < 3e-4
end # end test set
