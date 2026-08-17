# Fast Marching Method (a port of `M2DO_LSM/src/fast_marching_method.cpp`,
# adapted from Scikit-FMM). Solves the Eikonal equation to reinitialise a
# signed-distance function, or to extend boundary velocities through the
# narrow band. The `0` neighbour sentinel marks out-of-domain directions.

const doubleEpsilon = eps(Float64)
const maxDouble = floatmax(Float64)

"""
    FastMarchingMethod(mesh)

Solves the Eikonal equation by the fast marching method to reinitialise a
signed-distance function or to extend boundary velocities through the narrow
band. A port of `M2DO_LSM/src/fast_marching_method.cpp`, adapted from
Scikit-FMM. Use [`march!`](@ref) to run it.
"""
mutable struct FastMarchingMethod
    mesh::LevelSetMesh
    heap::Heap
    heapPtr::Vector{Int}
    nodeStatus::Vector{Int}
    signedDistanceCopy::Vector{Float64}
    sd::Vector{Float64}
    vel::Vector{Float64}
    isVelocity::Bool

    function FastMarchingMethod(mesh::LevelSetMesh)
        n = mesh.nNodes
        return new(
            mesh, Heap(0), zeros(Int, n), zeros(Int, n), zeros(n), zeros(n), zeros(n), false
        )
    end
end

"""
    march!(fmm, signedDistance)
    march!(fmm, signedDistance, velocity)

Run the fast marching method to reinitialize `signedDistance` to a signed
distance function, or (with `velocity`) to extend boundary velocities through
the narrow band without changing the signed distance.
"""
function march!(fmm::FastMarchingMethod, signedDistance::Vector{Float64})
    fmm.sd = signedDistance
    fmm.isVelocity = false
    initialise_frozen!(fmm)
    initialise_heap!(fmm)
    initialise_trial!(fmm)
    solve!(fmm)
    return signedDistance
end

# Extend boundary velocities through the narrow band without changing the
# signed distance.
function march!(
    fmm::FastMarchingMethod, signedDistance::Vector{Float64}, velocity::Vector{Float64}
)
    fmm.sd = signedDistance
    fmm.vel = velocity
    fmm.isVelocity = true
    initialise_frozen!(fmm)
    initialise_heap!(fmm)
    initialise_trial!(fmm)
    solve!(fmm)
    signedDistance .= fmm.signedDistanceCopy
    return velocity
end

function initialise_frozen!(fmm::FastMarchingMethod)
    mesh = fmm.mesh
    sd = fmm.sd
    sdcopy = fmm.signedDistanceCopy
    fill!(fmm.nodeStatus, FMM_NONE)
    copyto!(sdcopy, sd)
    for i in eachindex(mesh.nodes)
        if sd[i] == 0
            fmm.nodeStatus[i] = FMM_FROZEN
        end
    end
    for i in eachindex(mesh.nodes)
        if fmm.nodeStatus[i] == FMM_NONE
            dist = zeros(2)
            isBorder = false
            for j in 1:4
                neighbour = mesh.nodes[i].neighbours[j]
                if neighbour != 0 && sdcopy[i] * sdcopy[neighbour] < 0
                    isBorder = true
                    d = sdcopy[i] / (sdcopy[i] - sdcopy[neighbour])
                    dim = j < 3 ? 1 : 2
                    if dist[dim] == 0 || dist[dim] > d
                        dist[dim] = d
                    end
                end
            end
            if isBorder
                distSum = 0.0
                for j in 1:2
                    if dist[j] > 0
                        distSum += 1.0 / (dist[j] * dist[j])
                    end
                end
                sd[i] = sdcopy[i] < 0 ? -sqrt(1.0 / distSum) : sqrt(1.0 / distSum)
                fmm.nodeStatus[i] = FMM_FROZEN
            end
        end
    end
    return nothing
end

function initialise_heap!(fmm::FastMarchingMethod)
    nFar = count(==(FMM_NONE), fmm.nodeStatus)
    fmm.heap = Heap(nFar)
    return nothing
end

function initialise_trial!(fmm::FastMarchingMethod)
    mesh = fmm.mesh
    for i in eachindex(mesh.nodes)
        if fmm.nodeStatus[i] == FMM_NONE
            for j in 1:4
                neighbour = mesh.nodes[i].neighbours[j]
                if neighbour != 0 && (fmm.nodeStatus[neighbour] & FMM_FROZEN) != 0
                    if fmm.nodeStatus[i] == FMM_NONE
                        if fmm.isVelocity
                            if mesh.nodes[i].isActive
                                fmm.nodeStatus[i] = FMM_TRIAL
                                fmm.sd[i] = update_node(fmm, i)
                                fmm.heapPtr[i] = push!(fmm.heap, i, abs(fmm.sd[i]))
                            end
                        else
                            fmm.nodeStatus[i] = FMM_TRIAL
                            fmm.sd[i] = update_node(fmm, i)
                            fmm.heapPtr[i] = push!(fmm.heap, i, abs(fmm.sd[i]))
                        end
                    end
                end
            end
        end
    end
    return nothing
end

function solve!(fmm::FastMarchingMethod)
    mesh = fmm.mesh
    toFreeze = Vector{Int}(undef, mesh.nNodes)
    while !isempty(fmm.heap)
        addr, value = pop!(fmm.heap)
        fmm.nodeStatus[addr] = FMM_FROZEN
        fmm.isVelocity && finalise_velocity!(fmm, addr)
        toFreeze[1] = addr
        nFrozen = 1

        while !isempty(fmm.heap) && value == peek(fmm.heap)
            laddr, _ = pop!(fmm.heap)
            fmm.nodeStatus[laddr] = FMM_FROZEN
            fmm.isVelocity && finalise_velocity!(fmm, laddr)
            nFrozen += 1
            toFreeze[nFrozen] = laddr
        end

        for k in 1:nFrozen
            addr = toFreeze[k]
            for j in 1:4
                naddr = mesh.nodes[addr].neighbours[j]
                if naddr != 0 && fmm.nodeStatus[naddr] != FMM_FROZEN
                    d = update_node(fmm, naddr)
                    fmm.sd[naddr] = d
                    if (fmm.nodeStatus[naddr] & FMM_TRIAL) != 0
                        set_distance!(fmm.heap, fmm.heapPtr[naddr], abs(d))
                    elseif fmm.nodeStatus[naddr] == FMM_NONE
                        if !fmm.isVelocity || mesh.nodes[naddr].isActive
                            fmm.nodeStatus[naddr] = FMM_TRIAL
                            fmm.heapPtr[naddr] = push!(fmm.heap, naddr, abs(d))
                        end
                    end
                    naddr2 = mesh.nodes[naddr].neighbours[j]
                    if naddr2 != 0 && (fmm.nodeStatus[naddr2] & FMM_TRIAL) != 0
                        d = update_node(fmm, naddr2)
                        fmm.sd[naddr2] = d
                        set_distance!(fmm.heap, fmm.heapPtr[naddr2], abs(d))
                    end
                end
            end
        end
    end
    return nothing
end

function update_node(fmm::FastMarchingMethod, node::Int)
    mesh = fmm.mesh
    sd = fmm.sd
    aa = 9.0 / 4.0
    oneThird = 1.0 / 3.0
    a = 0.0
    b = 0.0
    c = 0.0

    for i in 0:1
        dist1 = maxDouble
        dist2 = maxDouble
        for j in 0:1
            index = 2 * i + j + 1
            n1 = mesh.nodes[node].neighbours[index]
            if n1 != 0 && (fmm.nodeStatus[n1] & FMM_FROZEN) != 0
                if abs(sd[n1]) < abs(dist1)
                    dist1 = sd[n1]
                    n2 = mesh.nodes[n1].neighbours[index]
                    if n2 != 0 && (fmm.nodeStatus[n2] & FMM_FROZEN) != 0
                        if abs(sd[n2]) <= abs(dist1)
                            dist2 = sd[n2]
                        end
                    end
                end
            end
        end
        if dist2 < maxDouble
            tp = oneThird * (4 * dist1 - dist2)
            a += aa
            b -= 2 * aa * tp
            c += aa * tp * tp
        elseif dist1 < maxDouble
            a += 1
            b -= 2 * dist1
            c += dist1 * dist1
        end
    end
    c -= 1
    return solve_quadratic(fmm, node, a, b, c)
end

function finalise_velocity!(fmm::FastMarchingMethod, node::Int)
    mesh = fmm.mesh
    sd = fmm.sd
    vel = fmm.vel
    dist = zeros(2)
    frontDist = zeros(2)
    veld = zeros(2)
    for i in 1:4
        dim = i < 3 ? 1 : 2
        neighbour = mesh.nodes[node].neighbours[i]
        if neighbour != 0 && (fmm.nodeStatus[neighbour] & FMM_FROZEN) != 0
            d = abs(sd[neighbour])
            if frontDist[dim] == 0 || frontDist[dim] > d
                frontDist[dim] = d
                d = sd[node] - sd[neighbour]
                dist[dim] = abs(d)
                veld[dim] = vel[neighbour]
            end
        end
    end
    numerator = 0.0
    denominator = 0.0
    for i in 1:2
        numerator += dist[i] * veld[i]
        denominator += dist[i]
    end
    denominator != 0 || error("Divide by zero error.")
    vel[node] = numerator / denominator
    return nothing
end

function solve_quadratic(
    fmm::FastMarchingMethod, node::Int, a::Float64, b::Float64, c::Float64
)
    discrim = b * b - 4 * a * c
    if discrim > 0
        r0 = (-b + sqrt(discrim)) / (2 * a)
        r1 = (-b - sqrt(discrim)) / (2 * a)
    else
        return fmm.sd[node]
    end
    return fmm.signedDistanceCopy[node] > doubleEpsilon ? r0 : r1
end
