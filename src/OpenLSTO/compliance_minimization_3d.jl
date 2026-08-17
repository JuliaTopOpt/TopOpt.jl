# 3D level-set compliance minimization (a port of
# `projects/3d/comp_min.cpp`). Minimizes compliance subject to a volume
# constraint on a cantilever (left face fixed, downward line load on the
# bottom edge of the right face).

# Optimum boundary-point velocities (a port of `PerformOptimization` in
# `M2DO_3D_LSM/lsm_opti_3d.cpp`, `algo == 0` Newton-Raphson branch).
function perform_optimization_3d(
    boundary_pts_one_vector::Vector{Float64},
    boundary_areas::Vector{Float64},
    bsens::Vector{Float64},
    vsens::Vector{Float64},
    max_vol::Real,
    move_limit::Real,
    nx::Real,
    ny::Real,
    nz::Real,
    volume_fractions::Vector{Float64},
)
    bpointsize = length(bsens)
    vol = sum(volume_fractions)
    abssens = maximum(abs, bsens)
    sf = -bsens ./ abssens
    sg = vsens
    cg = sg .* boundary_areas

    lambda_g = move_limit
    percent_vol = 0.5
    target_vol = vol + percent_vol * lambda_g * sum(cg)
    target_vol = max(max_vol * nx * ny * nz / 100.0, target_vol)

    domain_distance = zeros(bpointsize)
    for i in 1:bpointsize
        x = boundary_pts_one_vector[3i - 2]
        y = boundary_pts_one_vector[3i - 1]
        z = boundary_pts_one_vector[3i]
        domdist = min(
            abs(x - 0.0), abs(x - nx), abs(y - 0.0), abs(y - ny), abs(z - 0.0), abs(z - nz)
        )
        if x - nx >= 0 ||
            -(x - 0) >= 0 ||
            y - ny >= 0 ||
            -(y - 0) >= 0 ||
            z - nz >= 0 ||
            -(z - 0) >= 0
            domdist = -domdist
        end
        domain_distance[i] = min(domdist, move_limit)
    end

    function new_vol(lambda_cur)
        v = vol
        for i in 1:bpointsize
            v -= cg[i] * min(domain_distance[i], lambda_g * sg[i] + lambda_cur * sf[i])
        end
        return v
    end

    lambda_0 = 0.0
    delta_lambda = 0.01
    for _ in 1:50
        new_vol0 = new_vol(lambda_0)
        new_vol2 = new_vol(lambda_0 + delta_lambda)
        new_vol1 = new_vol(lambda_0 - delta_lambda)
        slope = (new_vol2 - new_vol1) / 2 / delta_lambda
        lambda_0 -= (new_vol0 - target_vol) / slope
        abs(new_vol0 - target_vol) / target_vol < 1.0e-3 && break
    end
    lambda_f = lambda_0

    opt_vel = zeros(bpointsize)
    abs_vel = 0.0
    for i in 1:bpointsize
        opt_vel[i] = min(lambda_f * sf[i] + lambda_g * sg[i], domain_distance[i])
        opt_vel[i] = clamp(opt_vel[i], -move_limit, move_limit)
        abs_vel = max(abs_vel, abs(opt_vel[i]))
    end
    if abs_vel > move_limit
        for i in 1:bpointsize
            opt_vel[i] = move_limit * opt_vel[i] / abs_vel
        end
    end
    return opt_vel
end

"""
    compliance_minimization_3d(; nelx, nely, nelz, ...)

Run the OpenLSTO 3D level-set compliance-minimization loop on a cantilever
(left face fixed, downward line load on the bottom edge of the right face).
Returns a `NamedTuple` with the final `level_set` ([`LevelSet3D`](@ref)), the
[`HexStudy`](@ref), the boundary conditions, and the per-iteration
`compliances` and volume-fraction `areas`.

Like the upstream `projects/3d/comp_min.cpp`, only the load region is pinned
solid by default (`pin_support=:none`). To also keep material on the left-face
supports — which the optimizer can otherwise erode onto void — pass
`pin_support=:soft` (a large support sensitivity, analogous to the load pin)
or `pin_support=:hard` (clamp the signed distance on the support face so the
contour cannot recede past it).
"""
function compliance_minimization_3d(;
    nelx::Integer=40,
    nely::Integer=20,
    nelz::Integer=20,
    E::Real=1.0,
    nu::Real=0.3,
    rho::Real=1.0,
    holes::Vector{Vector{Float64}}=Vector{Float64}[],
    max_iterations::Integer=50,
    max_vol::Real=30.0,
    move_limit::Real=0.25,
    pin_support::Symbol=:none,
    verbose::Bool=true,
)
    nelx = Int(nelx)
    nely = Int(nely)
    nelz = Int(nelz)

    material = HexMaterial(E, nu, rho)
    fixed_dofs = Int[]
    for y in 0:nely
        for z in 0:nelz
            node = _hex_node(nelx, nely, 0, y, z)
            push!(fixed_dofs, 3node - 2, 3node - 1, 3node)
        end
    end
    study = HexStudy(nelx, nely, nelz, material, fixed_dofs)

    load_dofs = Int[]
    load_values = Float64[]
    load_nodes = Tuple{Int,Vector{Float64}}[]
    for y in 0:nely
        node = _hex_node(nelx, nely, nelx, y, 0)
        append!(load_dofs, [3node - 2, 3node - 1, 3node])
        append!(load_values, [0.0, 0.0, -1.0])
        push!(load_nodes, (node, [0.0, 0.0, -1.0]))
    end
    assemble_hex_f!(study, load_dofs, load_values)
    boundary_conditions = LevelSetBoundaryConditions(
        load_nodes, _supports_from_fixed_dofs(fixed_dofs, 3)
    )

    lsm = LevelSet3D(nelx, nely, nelz; holes)
    lsm.boundary_conditions = boundary_conditions
    if pin_support == :hard
        fix_solid_face!(lsm, :x, 0)
    end

    compliances = Float64[]
    areas = Float64[]

    for iteration in 1:max_iterations
        marching_cubes_wrapper!(lsm)
        setup_narrow_band!(lsm)
        calculate_volume_fractions!(lsm)

        study.area_fractions .= max.(lsm.volumefraction_vector, 1e-3)
        assemble_hex_K!(study)
        solve_hex!(study)
        compliance = compute_hex_compliance_sensitivities!(study)
        push!(compliances, compliance)
        push!(areas, sum(lsm.volumefraction_vector) / (nelx * nely * nelz))

        bsens = Float64[]
        vsens = Float64[]
        for i in 1:(lsm.num_boundary_pts)
            bp = [
                lsm.boundary_pts_one_vector[3i - 2],
                lsm.boundary_pts_one_vector[3i - 1],
                lsm.boundary_pts_one_vector[3i],
            ]
            s = hex_boundary_sensitivity(study, bp)
            bs = -s
            # Pin the load (bottom edge of the right face) so the boundary
            # does not erode material from it.
            if bp[1] >= nelx - 2 && bp[3] <= 2
                bs = 1.0e5
            elseif pin_support == :soft && bp[1] <= 2
                bs = 1.0e5
            end
            push!(bsens, bs)
            push!(vsens, -1.0)
        end

        opt_vel = perform_optimization_3d(
            lsm.boundary_pts_one_vector,
            lsm.boundary_areas,
            bsens,
            vsens,
            max_vol,
            move_limit,
            nelx,
            nely,
            nelz,
            lsm.volumefraction_vector,
        )
        lsm.opt_vel = opt_vel

        extrapolate_velocities!(lsm)

        fast_marching_method!(lsm, lsm.indices_considered_inside)
        lsm.phi_temp .= -lsm.phi_temp
        fast_marching_method!(lsm, lsm.indices_considered_outside)
        lsm.phi_temp .= -lsm.phi_temp

        advect!(lsm)

        verbose && println(
            rpad(iteration, 4),
            " compliance=",
            round(compliance; digits=4),
            " volfrac=",
            round(areas[end]; digits=4),
        )
    end

    return (
        level_set=lsm,
        study=study,
        boundary_conditions=boundary_conditions,
        compliances=compliances,
        areas=areas,
    )
end
