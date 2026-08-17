# Top-level compliance-minimization loop (a port of
# `projects/compliance/main.cpp`). Solves: minimise compliance subject to a
# volume fraction, using the OpenLSTO level-set method.
#
# The compliance and volume objectives have differentiable counterparts
# elsewhere in TopOpt.jl: `ComplianceFun` (src/Functions/compliance.jl) and
# `VolumeFun` (src/Functions/volume.jl). OpenLSTO writes per-iteration VTK
# files; TopOpt.jl's equivalent output path is `save_mesh`/`visualize`.

"""
    compliance_minimization(; nelx, nely, ...)

Run the OpenLSTO level-set compliance-minimization loop on the cantilever
problem (left edge fixed, downward point load at the midpoint of the right
edge). Returns a `NamedTuple` with the final `level_set`, `boundary`, `study`,
`sensitivities`, and the per-iteration `compliances` and `areas`.

Keyword arguments:
  - `nelx`, `nely`: number of finite elements (and level-set cells) in x and y.
  - `E`, `nu`, `rho`: material properties (Young's modulus, Poisson's ratio,
    density).
  - `holes`: initial circular holes; defaults to the Swiss-cheese arrangement.
  - `move_limit`, `band_width`, `is_fixed_domain`: level-set parameters.
  - `max_iterations`, `max_area`, `max_diff`: stopping criteria.

# Example (the OpenLSTO cantilever demo)
holes = [Hole(16, 14, 5), Hole(48, 14, 5), Hole(80, 14, 5), Hole(112, 14, 5),
    Hole(144, 14, 5), Hole(32, 27, 5), Hole(64, 27, 5), Hole(96, 27, 5),
    Hole(128, 27, 5), Hole(16, 40, 5), Hole(48, 40, 5), Hole(80, 40, 5),
    Hole(112, 40, 5), Hole(144, 40, 5), Hole(32, 53, 5), Hole(64, 53, 5),
    Hole(96, 53, 5), Hole(128, 53, 5), Hole(16, 66, 5), Hole(48, 66, 5),
    Hole(80, 66, 5), Hole(112, 66, 5), Hole(144, 66, 5)]
result = compliance_minimization(; nelx=160, nely=80, holes)
"""
function compliance_minimization(;
    nelx::Integer=160,
    nely::Integer=80,
    E::Real=1.0,
    nu::Real=0.3,
    rho::Real=1.0,
    holes::Union{Nothing,Vector{Hole}}=nothing,
    move_limit::Real=0.5,
    band_width::Integer=6,
    is_fixed_domain::Bool=false,
    max_iterations::Integer=300,
    max_area::Real=0.5,
    max_diff::Real=1e-4,
    verbose::Bool=true,
)
    nelx = Int(nelx)
    nely = Int(nely)

    # Finite element setup.
    fea_mesh = FEMesh(nelx, nely)
    material = SolidMaterial(E, nu, rho)
    w = nelx + 1
    fixed_dofs = Int[]
    for y in 0:nely
        node = y * w + 1
        push!(fixed_dofs, 2node - 1, 2node)
    end
    study = StationaryStudy(fea_mesh, material, fixed_dofs)
    sens = SensitivityAnalysis(study)

    load_node = (nely ÷ 2) * w + nelx + 1
    assemble_f!(study, [2load_node - 1, 2load_node], [0.0, -0.5])

    # Level-set setup.
    lsm_mesh = Mesh(nelx, nely)
    mesh_area = Float64(lsm_mesh.width) * Float64(lsm_mesh.height)
    level_set = if holes === nothing
        LevelSet(lsm_mesh, move_limit, band_width, is_fixed_domain)
    else
        LevelSet(lsm_mesh, holes, move_limit, band_width, is_fixed_domain)
    end
    reinitialise!(level_set)
    boundary = Boundary(level_set)

    n_reinit = 0
    n_iterations = 0
    objective_values = Float64[]
    compliances = Float64[]
    areas = Float64[]
    relative_difference = 1.0

    verbose && println("--------------------------------")
    verbose && println(rpad("Iteration", 10), rpad("Compliance", 14), "Area")
    verbose && println("--------------------------------")

    while n_iterations < max_iterations
        n_iterations += 1

        discretise!(boundary, 2)
        compute_area_fractions!(boundary)

        area_fractions = Float64[]
        for element in lsm_mesh.elements
            push!(area_fractions, element.area < 1e-3 ? 1e-3 : element.area)
        end

        set_area_fractions!(study, area_fractions)
        assemble_K_with_area_fractions!(study)
        solve!(study)
        compute_compliance_sensitivities!(sens)

        for i in eachindex(boundary.points)
            boundary_point = [boundary.points[i].coord.x, boundary.points[i].coord.y]
            bsens = compute_boundary_sensitivity(sens, boundary_point)
            boundary.points[i].sensitivities[1] = -bsens
            boundary.points[i].sensitivities[2] = -1.0
        end

        optimise = Optimise(boundary.points, move_limit)
        optimise.length_x = Float64(lsm_mesh.width)
        optimise.length_y = Float64(lsm_mesh.height)
        optimise.boundary_area = boundary.area
        optimise.mesh_area = mesh_area
        optimise.max_area = Float64(max_area)
        solve_with_newton_raphson!(optimise)
        time_step = optimise.timeStep

        compute_velocities!(level_set, boundary.points)
        compute_gradients!(level_set)
        is_reinitialised = update!(level_set, time_step)

        if !is_reinitialised
            if n_reinit == 20
                reinitialise!(level_set)
                n_reinit = 0
            end
        else
            n_reinit = 0
        end
        n_reinit += 1

        area = boundary.area / mesh_area
        push!(compliances, sens.objective)
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
            round(area; digits=4),
        )

        if relative_difference < max_diff && area < 1.001 * max_area
            break
        end
    end

    return (; level_set, boundary, study, sensitivities=sens, compliances, areas)
end
