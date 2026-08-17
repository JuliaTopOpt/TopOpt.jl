# Top-level compliance-minimization loop (a port of
# `projects/compliance/main.cpp`). Solves: minimise compliance subject to a
# volume fraction, using the OpenLSTO level-set method.
#
# The compliance and volume objectives have differentiable counterparts
# elsewhere in TopOpt.jl: `ComplianceFun` (src/Functions/compliance.jl) and
# `VolumeFun` (src/Functions/volume.jl). OpenLSTO writes per-iteration VTK
# files; TopOpt.jl's equivalent output path is `save_mesh`/`visualize`.

"""
    LevelSetResult

The result of [`compliance_minimization`](@ref) or
[`stress_minimization`](@ref). Holds the final level set, its discretized
boundary, the finite element study and sensitivities, and the per-iteration
objective and area-fraction histories.

Fields:
- `level_set`: the final [`LevelSet`](@ref) (signed distance on the grid).
- `boundary`: the final [`LevelSetBoundary`](@ref) discretization.
- `study`: the [`StationaryStudy`](@ref) holding the last FEA solve.
- `sensitivities`: the [`SensitivityAnalysis`](@ref) from the last iteration.
- `objectives`: per-iteration objective (compliance or p-norm stress) values.
- `areas`: per-iteration volume-fraction values.
"""
struct LevelSetResult
    level_set::LevelSet
    boundary::LevelSetBoundary
    study::StationaryStudy
    sensitivities::SensitivityAnalysis
    objectives::Vector{Float64}
    areas::Vector{Float64}
end

"""
    area_fractions(result::LevelSetResult)

Re-discretize the final level set and return the per-cell material area
fraction, i.e. the density field of the optimized design. The vector has one
entry per cell, ordered row-major (x fastest) to match the finite element
mesh of `result.study`.
"""
function area_fractions(result::LevelSetResult)
    boundary = LevelSetBoundary(result.level_set)
    discretise!(boundary, 2)
    compute_area_fractions!(boundary)
    return [element.area for element in result.level_set.mesh.elements]
end

"""
    compliance_minimization(; nelx, nely, ...)

Run the OpenLSTO level-set compliance-minimization loop on the cantilever
problem (left edge fixed, downward point load at the midpoint of the right
edge). Returns a [`LevelSetResult`](@ref) with the final `level_set`,
`boundary`, `study`, `sensitivities`, and the per-iteration `objectives`
and `areas`.

Keyword arguments:
  - `nelx`, `nely`: number of finite elements (and level-set cells) in x and y.
  - `E`, `nu`, `rho`: material properties (Young's modulus, Poisson's ratio,
    density).
  - `holes`: initial circular holes; defaults to the Swiss-cheese arrangement.
  - `move_limit`, `band_width`, `is_fixed_domain`: level-set parameters.
  - `max_iterations`, `max_area`, `max_diff`: stopping criteria.
  - `hole_nucleation`: enable the hole-nucleation scheme (a port of
    `projects/hole_creation`); `hole_cfl`, `hole_l_band`, and
    `new_hole_area_limit` tune it.

# Example (the OpenLSTO cantilever demo)
holes = [LevelSetHole(16, 14, 5), LevelSetHole(48, 14, 5), LevelSetHole(80, 14, 5), LevelSetHole(112, 14, 5),
    LevelSetHole(144, 14, 5), LevelSetHole(32, 27, 5), LevelSetHole(64, 27, 5), LevelSetHole(96, 27, 5),
    LevelSetHole(128, 27, 5), LevelSetHole(16, 40, 5), LevelSetHole(48, 40, 5), LevelSetHole(80, 40, 5),
    LevelSetHole(112, 40, 5), LevelSetHole(144, 40, 5), LevelSetHole(32, 53, 5), LevelSetHole(64, 53, 5),
    LevelSetHole(96, 53, 5), LevelSetHole(128, 53, 5), LevelSetHole(16, 66, 5), LevelSetHole(48, 66, 5),
    LevelSetHole(80, 66, 5), LevelSetHole(112, 66, 5), LevelSetHole(144, 66, 5)]
result = compliance_minimization(; nelx=160, nely=80, holes)
"""
function compliance_minimization(;
    nelx::Integer=160,
    nely::Integer=80,
    E::Real=1.0,
    nu::Real=0.3,
    rho::Real=1.0,
    holes::Union{Nothing,Vector{LevelSetHole}}=nothing,
    move_limit::Real=0.5,
    band_width::Integer=6,
    is_fixed_domain::Bool=false,
    max_iterations::Integer=300,
    max_area::Real=0.5,
    max_diff::Real=1e-4,
    hole_nucleation::Bool=false,
    hole_cfl::Real=0.15,
    hole_l_band::Real=2.0,
    new_hole_area_limit::Real=0.03,
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
    lsm_mesh = LevelSetMesh(nelx, nely)
    mesh_area = Float64(lsm_mesh.width) * Float64(lsm_mesh.height)
    level_set = if holes === nothing
        LevelSet(lsm_mesh, move_limit, band_width, is_fixed_domain)
    else
        LevelSet(lsm_mesh, holes, move_limit, band_width, is_fixed_domain)
    end
    reinitialise!(level_set)
    boundary = LevelSetBoundary(level_set)

    # Hole-nucleation state (only used when `hole_nucleation` is enabled).
    h = max(lsm_mesh.width / nelx, lsm_mesh.height / nely)
    h_bar = h
    h_flag = false
    h_lsf = zeros(lsm_mesh.nNodes)

    n_reinit = 0
    n_iterations = 0
    objective_values = Float64[]
    objectives = Float64[]
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

        optimise = LevelSetOptimizer(boundary.points, move_limit)
        optimise.length_x = Float64(lsm_mesh.width)
        optimise.length_y = Float64(lsm_mesh.height)
        optimise.boundary_area = boundary.area
        optimise.mesh_area = mesh_area
        optimise.max_area = Float64(max_area)
        solve_with_newton_raphson!(optimise)
        time_step = optimise.timeStep

        if hole_nucleation &&
            n_iterations > 5 &&
            (boundary.area / mesh_area) > 1.05 * max_area
            h_count, h_index, h_elem = hole_map(lsm_mesh, level_set, h, hole_l_band)

            h_nsens_temp = [zeros(2) for _ in 1:(lsm_mesh.nNodes)]
            for inode in 1:(lsm_mesh.nNodes)
                n_point = [lsm_mesh.nodes[inode].coord.x, lsm_mesh.nodes[inode].coord.y]
                h_nsens_temp[inode][1] = -compute_boundary_sensitivity(sens, n_point)
                h_nsens_temp[inode][2] = -1.0
            end

            h_esens = [zeros(2) for _ in 1:(lsm_mesh.nElements)]
            for iel in 1:(lsm_mesh.nElements)
                for ind in 1:4
                    inode = lsm_mesh.elements[iel].nodes[ind]
                    h_esens[iel][1] += 0.25 * h_nsens_temp[inode][1]
                    h_esens[iel][2] += 0.25 * h_nsens_temp[inode][2]
                end
            end

            h_nsens = [zeros(2) for _ in 1:(lsm_mesh.nNodes)]
            for iel in 1:(lsm_mesh.nElements)
                for ind in 1:4
                    inode = lsm_mesh.elements[iel].nodes[ind]
                    h_nsens[inode][1] += 0.25 * h_esens[iel][1]
                    h_nsens[inode][2] += 0.25 * h_esens[iel][2]
                end
            end

            lambdas = [-optimise.lambda_f, -optimise.lambda_g]

            if h_flag
                fill!(h_lsf, h_bar)
                get_h_lsf!(h_index, h_nsens, lambdas, h_lsf)
                h_flag = false
            else
                get_h_lsf!(h_index, h_nsens, lambdas, h_lsf)

                for inode in 1:(lsm_mesh.nNodes)
                    if h_index[inode] && h_lsf[inode] < 0
                        h_flag = true
                    end
                end

                area_h_lsf = hole_area_fractions(lsm_mesh, h_lsf)
                area_lsf = hole_area_fractions(lsm_mesh, level_set.signedDistance)

                if h_flag
                    hole_area_fraction = (mesh_area - area_h_lsf) / area_lsf
                    temp_min_h_lsf = 1.0
                    if hole_area_fraction > new_hole_area_limit
                        for inode in 1:(lsm_mesh.nNodes)
                            temp_min_h_lsf = min(temp_min_h_lsf, h_lsf[inode])
                        end
                    end
                    while hole_area_fraction > new_hole_area_limit
                        for inode in 1:(lsm_mesh.nNodes)
                            h_lsf[inode] -= 0.005 * temp_min_h_lsf
                        end
                        area_h_lsf = hole_area_fractions(lsm_mesh, h_lsf)
                        hole_area_fraction = (mesh_area - area_h_lsf) / area_lsf
                    end

                    for inode in 1:(lsm_mesh.nNodes)
                        if h_index[inode] && hole_area_fraction > 1e-3
                            if h_lsf[inode] <= h_bar &&
                                h_lsf[inode] < level_set.signedDistance[inode]
                                level_set.signedDistance[inode] = h_lsf[inode]
                            end
                        end
                    end
                end

                if h_flag
                    fmm = FastMarchingMethod(lsm_mesh)
                    march!(fmm, level_set.signedDistance)
                end
            end
        end

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
        push!(objectives, sens.objective)
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

    return LevelSetResult(level_set, boundary, study, sens, objectives, areas)
end
