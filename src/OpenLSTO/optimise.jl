# Newton-Raphson solve for the boundary-point velocities (a port of
# `M2DO_LSM/src/optimise.cpp`). Given the boundary-point sensitivities of the
# objective (`sensitivities[1]`) and the volume constraint
# (`sensitivities[2] = -1`), it solves for the Lagrange multiplier λ_f so the
# volume constraint is met, then applies the boundary points' normal
# velocities `v = -min(λ_f s_f + λ_g s_g, d)`.

mutable struct Optimise
    boundaryPoints::Vector{BoundaryPoint}
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

function Optimise(boundaryPoints::Vector{BoundaryPoint}, moveLimit::Real)
    return Optimise(
        boundaryPoints, Float64(moveLimit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    )
end

function solve_with_newton_raphson!(optimise::Optimise)
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
