# Newton-Raphson solve for the boundary-point velocities (a port of
# `M2DO_LSM/src/optimise.cpp`). Given the boundary-point sensitivities of the
# objective (`sensitivities[1]`) and the volume constraint
# (`sensitivities[2] = -1`), it solves for the Lagrange multiplier λ_f so the
# volume constraint is met, then applies the boundary points' normal
# velocities `v = -min(λ_f s_f + λ_g s_g, d)`.

"""
    LevelSetOptimizer(boundary_points, move_limit)

Solves for the boundary-point velocities that advance the level set while
satisfying the volume constraint. Uses a Newton-Raphson iteration on the
Lagrange multiplier (a port of `M2DO_LSM/src/optimise.cpp`).
"""
mutable struct LevelSetOptimizer
    boundaryPoints::Vector{LevelSetBoundaryPoint}
    moveLimit::Float64
    boundary_area::Float64
    mesh_area::Float64
    max_area::Float64
    length_x::Float64
    length_y::Float64
    lambda_f::Float64
    lambda_g::Float64
    timeStep::Float64
end

function LevelSetOptimizer(boundaryPoints::Vector{LevelSetBoundaryPoint}, moveLimit::Real)
    return LevelSetOptimizer(
        boundaryPoints, Float64(moveLimit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    )
end

function solve_with_newton_raphson!(optimise::LevelSetOptimizer)
    boundaryPoints = optimise.boundaryPoints
    bpointsize = length(boundaryPoints)
    optimise.timeStep = 1.0

    # Normalise the objective sensitivities.
    abssens = maximum(abs(bp.sensitivities[1]) for bp in boundaryPoints)
    for bp in boundaryPoints
        bp.sensitivities[1] /= abssens
    end

    optimise.lambda_g = optimise.moveLimit

    fraction_area = 0.5
    target_area = optimise.boundary_area
    for bp in boundaryPoints
        target_area += bp.length * fraction_area * (-optimise.lambda_g)
    end
    target_area = max(optimise.max_area * optimise.mesh_area, target_area)

    # Signed distance from each boundary point to the domain boundary.
    domain_distance = Vector{Float64}(undef, bpointsize)
    for i in 1:bpointsize
        curpt = boundaryPoints[i].coord
        domdist = min(
            abs(curpt.x - 0.0),
            abs(curpt.x - optimise.length_x),
            abs(curpt.y - 0.0),
            abs(curpt.y - optimise.length_y),
        )
        if curpt.x - optimise.length_x >= 0.0 ||
            -(curpt.x - 0.0) >= 0.0 ||
            curpt.y - optimise.length_y >= 0.0 ||
            -(curpt.y - 0.0) >= 0.0
            domdist = -domdist
        end
        domain_distance[i] = domdist
    end

    lambda_0 = 0.0
    delta_lambda = 0.1
    for _ in 1:50
        new_area0 = optimise.boundary_area
        new_area2 = optimise.boundary_area
        new_area1 = optimise.boundary_area
        for i in 1:bpointsize
            bp = boundaryPoints[i]
            d = domain_distance[i]
            new_area0 +=
                bp.length * min(
                    d,
                    optimise.lambda_g * bp.sensitivities[2] +
                    lambda_0 * bp.sensitivities[1],
                )
            new_area2 +=
                bp.length * min(
                    d,
                    optimise.lambda_g * bp.sensitivities[2] +
                    (lambda_0 + delta_lambda) * bp.sensitivities[1],
                )
            new_area1 +=
                bp.length * min(
                    d,
                    optimise.lambda_g * bp.sensitivities[2] +
                    (lambda_0 - delta_lambda) * bp.sensitivities[1],
                )
        end
        slope = (new_area2 - new_area1) / 2 / delta_lambda
        lambda_0 -= (new_area0 - target_area) / slope
        abs(new_area0 - target_area) < 1e-3 && break
    end
    optimise.lambda_f = lambda_0

    for i in 1:bpointsize
        bp = boundaryPoints[i]
        bp.velocity =
            -min(
                optimise.lambda_f * bp.sensitivities[1] +
                optimise.lambda_g * bp.sensitivities[2],
                domain_distance[i],
            )
    end

    # CFL: scale the velocities when the largest exceeds the move limit.
    absvel = maximum(bp.velocity for bp in boundaryPoints)
    if absvel > optimise.moveLimit
        for bp in boundaryPoints
            bp.velocity *= optimise.moveLimit / absvel
        end
    end
    return optimise
end

# Optimum boundary-point velocities for the L-beam stress problem
# (OpenLSTO's `Solve_LbeamStress_With_NewtonRaphson`). Uses a reduced move
# limit for the bounds and the L-beam inner-corner distance to clamp the
# velocities.
function solve_lbeam_stress_with_newton_raphson!(
    optimise::LevelSetOptimizer, reduced_move_limit::Real
)
    boundaryPoints = optimise.boundaryPoints
    bpointsize = length(boundaryPoints)
    optimise.timeStep = 1.0

    abssens = maximum(abs(bp.sensitivities[1]) for bp in boundaryPoints)
    for bp in boundaryPoints
        bp.sensitivities[1] /= abssens
    end

    optimise.lambda_g = Float64(reduced_move_limit)

    fraction_area = 0.25
    target_area = optimise.boundary_area
    for bp in boundaryPoints
        target_area += bp.length * fraction_area * (-optimise.lambda_g)
    end
    target_area = max(optimise.max_area * optimise.mesh_area, target_area)

    upper_bound = Vector{Float64}(undef, bpointsize)
    lower_bound = fill(-Float64(reduced_move_limit), bpointsize)
    for i in 1:bpointsize
        bp = boundaryPoints[i]
        domdist = min(
            abs(bp.coord.x - 0.0),
            abs(bp.coord.x - optimise.length_x),
            abs(bp.coord.y - 0.0),
            abs(bp.coord.y - optimise.length_y),
        )
        domdist1 = min(
            bp.coord.x - 0.4 * optimise.length_x,
            -bp.coord.x + optimise.length_x,
            bp.coord.y - 0.4 * optimise.length_y,
            -bp.coord.y + optimise.length_y,
        )
        domdist = min(-domdist1, domdist)
        upper_bound[i] = min(domdist, Float64(reduced_move_limit))
    end

    function new_area(lambda_cur)
        area = optimise.boundary_area
        for i in 1:bpointsize
            bp = boundaryPoints[i]
            z = min(
                upper_bound[i],
                optimise.lambda_g * bp.sensitivities[2] + lambda_cur * bp.sensitivities[1],
            )
            z = max(lower_bound[i], z)
            area += bp.length * z
        end
        return area
    end

    lambda_0 = 0.0
    delta_lambda = 0.1
    for _ in 1:250
        new_area0 = new_area(lambda_0)
        new_area2 = new_area(lambda_0 + delta_lambda)
        new_area1 = new_area(lambda_0 - delta_lambda)
        slope = (new_area2 - new_area1) / 2 / delta_lambda
        lambda_0 -= (new_area0 - target_area) / slope
        abs(new_area0 - target_area) / optimise.mesh_area < 1e-3 && break
    end
    optimise.lambda_f = lambda_0

    for i in 1:bpointsize
        bp = boundaryPoints[i]
        z = min(
            upper_bound[i],
            optimise.lambda_f * bp.sensitivities[1] +
            optimise.lambda_g * bp.sensitivities[2],
        )
        z = max(lower_bound[i], z)
        bp.velocity = -z
    end

    absvel = maximum(abs(bp.velocity) for bp in boundaryPoints)
    if absvel > optimise.moveLimit
        for bp in boundaryPoints
            bp.velocity *= optimise.moveLimit / absvel
        end
    end
    return optimise
end
