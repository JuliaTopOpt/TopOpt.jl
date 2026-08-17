# Stress minimization on an L-beam with the level-set method (a port of
# `projects/stress_min/lbeam.cpp`). Minimizes the p-norm of the von Mises
# stress subject to a volume constraint, using the adjoint-based stress
# sensitivity and the L-beam-specific velocity bounds.

"""
    stress_minimization(; nelx, nely, ...)

Run the OpenLSTO level-set stress-minimization loop on the L-beam problem
(top edge fixed, downward point load on the right edge at 2/5 height). The
objective is the p-norm of the von Mises stress; the sensitivities use the
adjoint method ([`compute_stress_sensitivities!`](@ref)). Returns a
[`LevelSetResult`](@ref) whose `objectives` field holds the per-iteration
p-norm stress.

Keyword arguments:
  - `nelx`, `nely`: number of finite elements (and level-set cells) in x and y.
  - `E`, `nu`, `rho`: material properties.
  - `holes`: initial circular holes; defaults to the five-hole L-beam seeding.
  - `move_limit`, `band_width`: level-set parameters.
  - `max_iterations`, `max_area`, `max_diff`, `p_norm`, `reduced_move_limit`:
    stopping and optimization parameters.
"""
function stress_minimization(;
    nelx::Integer=100,
    nely::Integer=100,
    E::Real=1.0,
    nu::Real=0.3,
    rho::Real=1.0,
    holes::Union{Nothing,Vector{LevelSetHole}}=nothing,
    move_limit::Real=0.5,
    band_width::Integer=6,
    max_iterations::Integer=500,
    max_area::Real=0.4,
    max_diff::Real=5e-4,
    p_norm::Real=6.0,
    least_sq_radius::Real=2.0,
    reduced_move_limit::Real=0.15,
    verbose::Bool=true,
)
    nelx = Int(nelx)
    nely = Int(nely)

    # Finite element setup: top edge fixed, point load on the right edge.
    fea_mesh = FEMesh(nelx, nely)
    material = SolidMaterial(E, nu, rho)
    w = nelx + 1
    fixed_dofs = Int[]
    for x in 0:nelx
        node = nely * w + x + 1
        push!(fixed_dofs, 2node - 1, 2node)
    end
    study = StationaryStudy(fea_mesh, material, fixed_dofs)
    sens = SensitivityAnalysis(study)

    load_y = round(Int, nely * 2 / 5)
    load_node = load_y * w + nelx + 1
    load_dofs = [2load_node - 1, 2load_node]
    load_values = [0.0, -3.0]
    boundary_conditions = LevelSetBoundaryConditions(
        [(load_node, [0.0, -3.0])], _supports_from_fixed_dofs(fixed_dofs, 2)
    )

    # Level-set setup: L-beam domain with an inner corner at 2/5 width.
    lsm_mesh = LevelSetMesh(nelx, nely)
    inner_corner = 2 / 5 * nelx
    vertical_edge = [
        Coord(inner_corner - 0.01, inner_corner - 0.01),
        Coord(inner_corner + 0.01, nely + 0.01),
    ]
    horizontal_edge = [
        Coord(inner_corner - 0.01, inner_corner - 0.01),
        Coord(nelx + 0.01, inner_corner + 0.01),
    ]
    create_mesh_boundary!(lsm_mesh, vertical_edge)
    create_mesh_boundary!(lsm_mesh, horizontal_edge)

    level_set = if holes === nothing
        LevelSet(
            lsm_mesh,
            LevelSetHole[
                LevelSetHole(20, 20, 10),
                LevelSetHole(20, 50, 10),
                LevelSetHole(20, 80, 10),
                LevelSetHole(50, 20, 10),
                LevelSetHole(80, 20, 10),
            ],
            move_limit,
            band_width,
            false,
        )
    else
        LevelSet(lsm_mesh, holes, move_limit, band_width, false)
    end
    kill_region = [
        Coord(inner_corner + 0.01, inner_corner + 0.01), Coord(nelx + 0.01, nely + 0.01)
    ]
    kill_nodes!(level_set, kill_region)
    create_level_set_boundary!(level_set, vertical_edge)
    create_level_set_boundary!(level_set, horizontal_edge)
    fix_points = [
        Coord(nelx - 3.01, 2 / 5 * nely - 2.01), Coord(nelx + 0.01, 2 / 5 * nely + 0.01)
    ]
    fix_nodes!(level_set, fix_points)
    reinitialise!(level_set)
    boundary = LevelSetBoundary(level_set)

    mesh_area = lsm_mesh.width * lsm_mesh.height - (3 / 5 * lsm_mesh.width)^2

    n_reinit = 0
    n_iterations = 0
    objective_values = Float64[]
    stresses = Float64[]
    areas = Float64[]
    relative_difference = 1.0

    verbose && println("---------------------------------------------")
    verbose &&
        println(rpad("Iteration", 10), rpad("Objective", 14), rpad("Tvm_max", 12), "Area")
    verbose && println("---------------------------------------------")

    while n_iterations < max_iterations
        n_iterations += 1

        discretise!(boundary, 2)
        compute_area_fractions!(boundary)

        area_fractions = Float64[]
        for element in lsm_mesh.elements
            push!(area_fractions, element.area < 1e-6 ? 1e-6 : element.area)
        end

        set_area_fractions!(study, area_fractions)
        assemble_K_with_area_fractions!(study)
        assemble_f!(study, load_dofs, load_values)
        solve!(study)
        compute_stress_sensitivities!(sens, p_norm)

        for i in eachindex(boundary.points)
            point = [boundary.points[i].coord.x, boundary.points[i].coord.y]
            bsens = compute_boundary_sensitivity(
                sens, point; radius=least_sq_radius, indicator=1, p_norm
            )
            boundary.points[i].sensitivities[1] = -bsens
            boundary.points[i].sensitivities[2] = -1.0
        end

        optimise = LevelSetOptimizer(boundary.points, move_limit)
        optimise.length_x = Float64(lsm_mesh.width)
        optimise.length_y = Float64(lsm_mesh.height)
        optimise.boundary_area = boundary.area
        optimise.mesh_area = mesh_area
        optimise.max_area = Float64(max_area)
        solve_lbeam_stress_with_newton_raphson!(optimise, reduced_move_limit)
        time_step = optimise.timeStep

        compute_velocities!(level_set, boundary.points)
        compute_gradients!(level_set)
        is_reinitialised = update!(level_set, time_step)

        if !is_reinitialised
            if n_reinit == 1
                reinitialise!(level_set)
                n_reinit = 0
            end
        else
            n_reinit = 0
        end
        n_reinit += 1

        area = boundary.area / mesh_area
        push!(stresses, sens.objective)
        push!(areas, area)
        push!(objective_values, sens.objective)

        if n_iterations > 5
            objective_value_k = sens.objective
            relative_difference = 0.0
            for i in 1:5
                objective_value_m = objective_values[n_iterations - i]
                relative_difference = max(
                    relative_difference,
                    abs((objective_value_k - objective_value_m) / objective_value_k),
                )
            end
        end

        verbose && println(
            rpad(string(n_iterations), 10),
            rpad(string(round(sens.objective; digits=4)), 14),
            rpad(string(round(sens.von_mises_max; digits=4)), 12),
            round(area; digits=4),
        )

        if relative_difference <= max_diff && area <= 1.001 * max_area
            break
        end
    end

    return LevelSetResult(
        level_set, boundary, study, sens, boundary_conditions, stresses, areas
    )
end
