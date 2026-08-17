# 3D level-set method (a port of `M2DO_3D_LSM/lsm_3d.cpp`). A signed-distance
# field on a structured grid is discretized with marching cubes and advanced
# by boundary velocities, with the narrow band reinitialized by an Eikonal
# (fast marching) solve.

"""
    LevelSet3D(nx, ny, nz)

A signed-distance level set on a 3D structured grid of `nx x ny x nz` cells.
Positive values are inside the structure. The zero surface is discretized
with marching cubes and advected by boundary velocities.
"""
mutable struct LevelSet3D
    nx::Int
    ny::Int
    nz::Int
    num_grid_pts::Int
    num_cells::Int
    narrow_band_width::Int
    num_gauss_pts::Int
    phi::Vector{Float64}
    phi_temp::Vector{Float64}
    phi_status::Vector{Int}
    grid_vel::Vector{Float64}
    grid_gradient::Vector{Float64}
    opt_vel::Vector{Float64}
    volumefraction_vector::Vector{Float64}
    boundary_pts::Vector{Vector{Float64}}
    boundary_pts_one_vector::Vector{Float64}
    boundary_areas::Vector{Float64}
    indices_considered_inside::Vector{Int}
    indices_considered_outside::Vector{Int}
    holes::Vector{Vector{Float64}}
    num_boundary_pts::Int
    num_triangles::Int
    triangles::Vector{NTuple{3,NTuple{3,Float64}}}
end

function LevelSet3D(
    nx::Integer,
    ny::Integer,
    nz::Integer;
    holes::Vector{Vector{Float64}}=Vector{Float64}[],
    narrow_band_width::Integer=3,
    num_gauss_pts::Integer=2,
)
    nx = Int(nx)
    ny = Int(ny)
    nz = Int(nz)
    num_grid_pts = (nx + 1) * (ny + 1) * (nz + 1)
    num_cells = nx * ny * nz
    lsm = LevelSet3D(
        nx,
        ny,
        nz,
        num_grid_pts,
        num_cells,
        Int(narrow_band_width),
        Int(num_gauss_pts),
        zeros(num_grid_pts),
        zeros(num_grid_pts),
        fill(-1, num_grid_pts),
        zeros(num_grid_pts),
        ones(num_grid_pts),
        Float64[],
        Float64[],
        Vector{Vector{Float64}}(),
        Float64[],
        Float64[],
        Int[],
        Int[],
        holes,
        0,
        0,
        NTuple{3,NTuple{3,Float64}}[],
    )
    make_box!(lsm)
    return lsm
end

# Initialize the signed distance to a box (minimum distance to the domain
# boundary, minus any seeded holes).
function make_box!(lsm::LevelSet3D)
    for x in 0:(lsm.nx)
        for y in 0:(lsm.ny)
            for z in 0:(lsm.nz)
                i = _idx3(lsm, x, y, z)
                val = min(x, lsm.nx - x, y, lsm.ny - y, z, lsm.nz - z)
                for hole in lsm.holes
                    dist2 = (hole[1] - x)^2 + (hole[2] - y)^2 + (hole[3] - z)^2 - hole[4]^2
                    dist = dist2 >= 0 ? sqrt(dist2) : -sqrt(-dist2)
                    val = min(val, dist)
                end
                lsm.phi[i] = val
            end
        end
    end
    return lsm
end

# Grid-point index (1-based) of (x, y, z), z-fastest (then y, then x).
function _idx3(lsm::LevelSet3D, x::Int, y::Int, z::Int)
    return z + y * (lsm.nz + 1) + x * (lsm.nz + 1) * (lsm.ny + 1) + 1
end

# 0-based grid coordinate (x, y, z) of a 1-based grid-point index.
function _grid_pt(lsm::LevelSet3D, index::Int)
    i0 = index - 1
    x = i0 ÷ ((lsm.nz + 1) * (lsm.ny + 1))
    y = (i0 ÷ (lsm.nz + 1)) % (lsm.ny + 1)
    z = i0 % (lsm.nz + 1)
    return (x, y, z)
end

# Discretize the zero surface with marching cubes, then store the triangle
# centroids (as boundary points) and areas (Heron's formula).
function marching_cubes_wrapper!(lsm::LevelSet3D)
    lsm.triangles = marching_cubes_3d(lsm.nx, lsm.ny, lsm.nz, lsm.phi)
    lsm.num_triangles = length(lsm.triangles)
    empty!(lsm.boundary_pts)
    empty!(lsm.boundary_pts_one_vector)
    empty!(lsm.boundary_areas)
    for tri in lsm.triangles
        p1, p2, p3 = tri
        cx = (p1[1] + p2[1] + p3[1]) / 3.0
        cy = (p1[2] + p2[2] + p3[2]) / 3.0
        cz = (p1[3] + p2[3] + p3[3]) / 3.0
        push!(lsm.boundary_pts, [cx, cy, cz])
        append!(lsm.boundary_pts_one_vector, cx, cy, cz)
        a = sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2 + (p1[3] - p2[3])^2)
        b = sqrt((p1[1] - p3[1])^2 + (p1[2] - p3[2])^2 + (p1[3] - p3[3])^2)
        c = sqrt((p3[1] - p2[1])^2 + (p3[2] - p2[2])^2 + (p3[3] - p2[3])^2)
        s = 0.5 * (a + b + c)
        push!(lsm.boundary_areas, sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0)))
    end
    lsm.num_boundary_pts = length(lsm.boundary_pts)
    return lsm
end

# Element volume fractions (xyz axis system, x-fastest), computed with Gauss
# points over the signed-distance field.
function calculate_volume_fractions!(lsm::LevelSet3D)
    resize!(lsm.volumefraction_vector, lsm.num_cells)
    ng = lsm.num_gauss_pts
    weights = fill(1.0 / ng, ng)
    gauss_points = [1.0 * (i + 1) / (1.0 + ng) for i in 0:(ng - 1)]
    for k in 0:(lsm.nz - 1)
        for j in 0:(lsm.ny - 1)
            for i in 0:(lsm.nx - 1)
                counter_cell = i + lsm.nx * j + lsm.nx * lsm.ny * k + 1
                idx = [
                    _idx3(lsm, i, j, k),
                    _idx3(lsm, i + 1, j, k),
                    _idx3(lsm, i, j + 1, k),
                    _idx3(lsm, i + 1, j + 1, k),
                    _idx3(lsm, i, j, k + 1),
                    _idx3(lsm, i + 1, j, k + 1),
                    _idx3(lsm, i, j + 1, k + 1),
                    _idx3(lsm, i + 1, j + 1, k + 1),
                ]
                if all(lsm.phi[idx[t]] >= 0 for t in 1:8)
                    lsm.volumefraction_vector[counter_cell] = 1.0
                elseif all(lsm.phi[idx[t]] < 0 for t in 1:8)
                    lsm.volumefraction_vector[counter_cell] = 0.0
                else
                    vf = 0.0
                    for gi in 1:ng, gj in 1:ng, gk in 1:ng
                        xl = gauss_points[gi]
                        yl = gauss_points[gj]
                        zl = gauss_points[gk]
                        up =
                            lsm.phi[idx[1]] * (1 - xl) * (1 - yl) * (1 - zl) +
                            lsm.phi[idx[2]] * xl * (1 - yl) * (1 - zl) +
                            lsm.phi[idx[3]] * (1 - xl) * yl * (1 - zl) +
                            lsm.phi[idx[4]] * xl * yl * (1 - zl) +
                            lsm.phi[idx[5]] * (1 - xl) * (1 - yl) * zl +
                            lsm.phi[idx[6]] * xl * (1 - yl) * zl +
                            lsm.phi[idx[7]] * (1 - xl) * yl * zl +
                            lsm.phi[idx[8]] * xl * yl * zl
                        u = 0.0
                        if up > 0.0
                            u = 1.0
                        end
                        if up <= 1.0 / ng && up >= 0.0
                            u = (1.0 * ng + 1.0) * up
                        end
                        vf += weights[gi] * weights[gj] * weights[gk] * u
                    end
                    lsm.volumefraction_vector[counter_cell] = vf
                end
            end
        end
    end
    return lsm
end

# Mark grid points in the narrow band and collect the "considered" points
# (inside and outside) for the fast marching reinitialization.
function setup_narrow_band!(lsm::LevelSet3D)
    fill!(lsm.phi_status, -1)
    lsm.phi_temp .= lsm.phi
    nb = lsm.narrow_band_width

    for i1 in 1:(lsm.num_boundary_pts)
        cp = lsm.boundary_pts[i1]
        for i in (1 - nb):(1 + nb), j in (1 - nb):(1 + nb), k in (1 - nb):(1 + nb)
            x_index = floor(Int, cp[1] + 0.5) + i
            y_index = floor(Int, cp[2] + 0.5) + j
            z_index = floor(Int, cp[3] + 0.5) + k
            if x_index > 0 &&
                y_index > 0 &&
                z_index > 0 &&
                x_index < lsm.nx + 2 &&
                y_index < lsm.ny + 2 &&
                z_index < lsm.nz + 2
                if max(
                    abs(x_index - 1 - cp[1]),
                    abs(y_index - 1 - cp[2]),
                    abs(z_index - 1 - cp[3]),
                ) < 1.0001
                    lsm.phi_status[_idx3(lsm, x_index - 1, y_index - 1, z_index - 1)] = 1
                end
            end
        end
    end

    empty!(lsm.indices_considered_inside)
    empty!(lsm.indices_considered_outside)
    for i1 in 1:(lsm.num_boundary_pts)
        cp = lsm.boundary_pts[i1]
        for i in (1 - nb):(1 + nb), j in (1 - nb):(1 + nb), k in (1 - nb):(1 + nb)
            x_index = floor(Int, cp[1] + 0.5) + i
            y_index = floor(Int, cp[2] + 0.5) + j
            z_index = floor(Int, cp[3] + 0.5) + k
            if max(
                abs(x_index - 1 - cp[1]), abs(y_index - 1 - cp[2]), abs(z_index - 1 - cp[3])
            ) <= nb
                if x_index > 0 &&
                    y_index > 0 &&
                    z_index > 0 &&
                    x_index < lsm.nx + 2 &&
                    y_index < lsm.ny + 2 &&
                    z_index < lsm.nz + 2
                    gi = _idx3(lsm, x_index - 1, y_index - 1, z_index - 1)
                    if lsm.phi_status[gi] < 1
                        lsm.phi_status[gi] = 2
                        if lsm.phi[gi] >= 0
                            push!(lsm.indices_considered_inside, gi)
                        else
                            push!(lsm.indices_considered_outside, gi)
                        end
                    end
                end
            end
        end
    end
    return lsm
end

# Extrapolate boundary-point velocities to the grid by inverse-square
# distance weighting.
function extrapolate_velocities!(lsm::LevelSet3D)
    weight = zeros(lsm.num_grid_pts)
    weightedvel = zeros(lsm.num_grid_pts)
    fill!(lsm.grid_vel, 0.0)
    vel_band_width = 2
    for i1 in 1:(lsm.num_boundary_pts)
        cp = lsm.boundary_pts[i1]
        for i in (1 - vel_band_width):(1 + vel_band_width),
            j in (1 - vel_band_width):(1 + vel_band_width),
            k in (1 - vel_band_width):(1 + vel_band_width)

            x_index = floor(Int, cp[1] + 0.5) + i
            y_index = floor(Int, cp[2] + 0.5) + j
            z_index = floor(Int, cp[3] + 0.5) + k
            if x_index > 0 &&
                y_index > 0 &&
                z_index > 0 &&
                x_index < lsm.nx + 2 &&
                y_index < lsm.ny + 2 &&
                z_index < lsm.nz + 2
                dist = sqrt(
                    (x_index - 1 - cp[1])^2 +
                    (y_index - 1 - cp[2])^2 +
                    (z_index - 1 - cp[3])^2,
                )
                weight_temp = 1.0 / max(dist, 1.0e-6)
                weight_temp *= weight_temp
                idx = _idx3(lsm, x_index - 1, y_index - 1, z_index - 1)
                weightedvel[idx] += weight_temp * lsm.opt_vel[i1]
                weight[idx] += weight_temp
            end
        end
    end
    for i in 1:(lsm.num_grid_pts)
        if weight[i] > 0
            lsm.grid_vel[i] = weightedvel[i] / weight[i]
        end
    end
    return lsm
end

# Solve the Eikonal equation at a grid point (fast marching update).
function solve_eikonal!(lsm::LevelSet3D, x::Int, y::Int, z::Int)
    phix = if x == 0
        lsm.phi_temp[_idx3(lsm, x + 1, y, z)]
    elseif x == lsm.nx
        lsm.phi_temp[_idx3(lsm, x - 1, y, z)]
    else
        min(lsm.phi_temp[_idx3(lsm, x + 1, y, z)], lsm.phi_temp[_idx3(lsm, x - 1, y, z)])
    end
    phiy = if y == 0
        lsm.phi_temp[_idx3(lsm, x, y + 1, z)]
    elseif y == lsm.ny
        lsm.phi_temp[_idx3(lsm, x, y - 1, z)]
    else
        min(lsm.phi_temp[_idx3(lsm, x, y + 1, z)], lsm.phi_temp[_idx3(lsm, x, y - 1, z)])
    end
    phiz = if z == 0
        lsm.phi_temp[_idx3(lsm, x, y, z + 1)]
    elseif z == lsm.nz
        lsm.phi_temp[_idx3(lsm, x, y, z - 1)]
    else
        min(lsm.phi_temp[_idx3(lsm, x, y, z + 1)], lsm.phi_temp[_idx3(lsm, x, y, z - 1)])
    end
    a_quad = 3.0
    b_quad = -2.0 * (phix + phiy + phiz)
    c_quad = phix * phix + phiy * phiy + phiz * phiz - 1.0
    if b_quad * b_quad >= 4 * a_quad * c_quad
        lsm.phi_temp[_idx3(lsm, x, y, z)] =
            0.5 * (-b_quad + sqrt(b_quad * b_quad - 4 * a_quad * c_quad)) / a_quad
    else
        lsm.phi_temp[_idx3(lsm, x, y, z)] = min(phix, phiy, phiz) + 0.75
    end
    return lsm
end

# Update the velocity at a grid point from its nearest accepted neighbors.
function update_velocity!(lsm::LevelSet3D, x::Int, y::Int, z::Int)
    # x direction
    if x == 0
        ext_weight_x =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x + 1, y, z)]
        ext_vel_x = lsm.grid_vel[_idx3(lsm, x + 1, y, z)]
    elseif x == lsm.nx
        ext_weight_x =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x - 1, y, z)]
        ext_vel_x = lsm.grid_vel[_idx3(lsm, x - 1, y, z)]
    elseif lsm.phi_temp[_idx3(lsm, x - 1, y, z)] < lsm.phi_temp[_idx3(lsm, x + 1, y, z)]
        ext_weight_x =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x - 1, y, z)]
        ext_vel_x = lsm.grid_vel[_idx3(lsm, x - 1, y, z)]
    else
        ext_weight_x =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x + 1, y, z)]
        ext_vel_x = lsm.grid_vel[_idx3(lsm, x + 1, y, z)]
    end
    # y direction
    if y == 0
        ext_weight_y =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y + 1, z)]
        ext_vel_y = lsm.grid_vel[_idx3(lsm, x, y + 1, z)]
    elseif y == lsm.ny
        ext_weight_y =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y - 1, z)]
        ext_vel_y = lsm.grid_vel[_idx3(lsm, x, y - 1, z)]
    elseif lsm.phi_temp[_idx3(lsm, x, y - 1, z)] < lsm.phi_temp[_idx3(lsm, x, y + 1, z)]
        ext_weight_y =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y - 1, z)]
        ext_vel_y = lsm.grid_vel[_idx3(lsm, x, y - 1, z)]
    else
        ext_weight_y =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y + 1, z)]
        ext_vel_y = lsm.grid_vel[_idx3(lsm, x, y + 1, z)]
    end
    # z direction
    if z == 0
        ext_weight_z =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y, z + 1)]
        ext_vel_z = lsm.grid_vel[_idx3(lsm, x, y, z + 1)]
    elseif z == lsm.nz
        ext_weight_z =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y, z - 1)]
        ext_vel_z = lsm.grid_vel[_idx3(lsm, x, y, z - 1)]
    elseif lsm.phi_temp[_idx3(lsm, x, y, z - 1)] < lsm.phi_temp[_idx3(lsm, x, y, z + 1)]
        ext_weight_z =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y, z - 1)]
        ext_vel_z = lsm.grid_vel[_idx3(lsm, x, y, z - 1)]
    else
        ext_weight_z =
            lsm.phi_temp[_idx3(lsm, x, y, z)] - lsm.phi_temp[_idx3(lsm, x, y, z + 1)]
        ext_vel_z = lsm.grid_vel[_idx3(lsm, x, y, z + 1)]
    end
    ext_weight_x = max(1.0e-6, ext_weight_x)
    ext_weight_y = max(1.0e-6, ext_weight_y)
    ext_weight_z = max(1.0e-6, ext_weight_z)
    gridvel =
        (ext_vel_x * ext_weight_x + ext_vel_y * ext_weight_y + ext_vel_z * ext_weight_z) /
        (ext_weight_x + ext_weight_y + ext_weight_z)
    gridvel = clamp(gridvel, -1.0, 1.0)
    lsm.grid_vel[_idx3(lsm, x, y, z)] = gridvel
    return lsm
end

# Fast marching: solve the Eikonal equation for the considered points and
# extend the velocities through the narrow band.
function fast_marching_method!(lsm::LevelSet3D, considered::Vector{Int})
    for idx in considered
        x, y, z = _grid_pt(lsm, idx)
        solve_eikonal!(lsm, x, y, z)
    end
    perm = sortperm(considered; by=idx -> lsm.phi_temp[idx])
    for idx in considered[perm]
        x, y, z = _grid_pt(lsm, idx)
        solve_eikonal!(lsm, x, y, z)
        update_velocity!(lsm, x, y, z)
    end
    return lsm
end

# 5th-order Hamilton-Jacobi WENO minus/plus stencils along an axis.
function _weno_minus(ph::Function, p::Int, n::Int)
    if p == 0
        v5 = ph(p + 2) - ph(p + 1)
        v4 = ph(p + 1) - ph(p)
        v3 = v2 = v1 = v4
    elseif p == 1
        v5 = ph(p + 2) - ph(p + 1)
        v4 = ph(p + 1) - ph(p)
        v3 = ph(p) - ph(p - 1)
        v2 = v1 = v3
    elseif p == 2
        v5 = ph(p + 2) - ph(p + 1)
        v4 = ph(p + 1) - ph(p)
        v3 = ph(p) - ph(p - 1)
        v2 = ph(p - 1) - ph(p - 2)
        v1 = v2
    elseif p <= n - 2
        v5 = ph(p + 2) - ph(p + 1)
        v4 = ph(p + 1) - ph(p)
        v3 = ph(p) - ph(p - 1)
        v2 = ph(p - 1) - ph(p - 2)
        v1 = ph(p - 2) - ph(p - 3)
    elseif p == n - 1
        v4 = ph(p + 1) - ph(p)
        v3 = ph(p) - ph(p - 1)
        v2 = ph(p - 1) - ph(p - 2)
        v1 = ph(p - 2) - ph(p - 3)
        v5 = v4
    else
        v3 = ph(p) - ph(p - 1)
        v2 = ph(p - 1) - ph(p - 2)
        v1 = ph(p - 2) - ph(p - 3)
        v5 = v4 = v3
    end
    return grad_HJ_WENO(v1, v2, v3, v4, v5)
end

function _weno_plus(ph::Function, p::Int, n::Int)
    if p == 0
        v1 = ph(p + 3) - ph(p + 2)
        v2 = ph(p + 2) - ph(p + 1)
        v3 = ph(p + 1) - ph(p)
        v4 = v5 = v3
    elseif p == 1
        v1 = ph(p + 3) - ph(p + 2)
        v2 = ph(p + 2) - ph(p + 1)
        v3 = ph(p + 1) - ph(p)
        v4 = ph(p) - ph(p - 1)
        v5 = v4
    elseif p <= n - 3
        v1 = ph(p + 3) - ph(p + 2)
        v2 = ph(p + 2) - ph(p + 1)
        v3 = ph(p + 1) - ph(p)
        v4 = ph(p) - ph(p - 1)
        v5 = ph(p - 1) - ph(p - 2)
    elseif p == n - 2
        v2 = ph(p + 2) - ph(p + 1)
        v3 = ph(p + 1) - ph(p)
        v4 = ph(p) - ph(p - 1)
        v5 = ph(p - 1) - ph(p - 2)
        v1 = v2
    elseif p == n - 1
        v3 = ph(p + 1) - ph(p)
        v4 = ph(p) - ph(p - 1)
        v5 = ph(p - 1) - ph(p - 2)
        v1 = v2 = v3
    else
        v4 = ph(p) - ph(p - 1)
        v5 = ph(p - 1) - ph(p - 2)
        v1 = v2 = v3 = v4
    end
    return grad_HJ_WENO(v1, v2, v3, v4, v5)
end

# Compute the WENO gradient of the signed distance within the narrow band.
function compute_gradients!(lsm::LevelSet3D)
    fill!(lsm.grid_gradient, 1.0)
    for i1 in 1:(lsm.num_grid_pts)
        lsm.phi_status[i1] == 1 || continue
        sign = lsm.grid_vel[i1] < 0 ? -1.0 : 1.0
        x, y, z = _grid_pt(lsm, i1)
        ph_x = (p) -> lsm.phi_temp[_idx3(lsm, p, y, z)]
        ph_y = (p) -> lsm.phi_temp[_idx3(lsm, x, p, z)]
        ph_z = (p) -> lsm.phi_temp[_idx3(lsm, x, y, p)]
        grad_x_minus = sign * _weno_minus(ph_x, x, lsm.nx)
        grad_x_plus = sign * _weno_plus(ph_x, x, lsm.nx)
        grad_y_minus = sign * _weno_minus(ph_y, y, lsm.ny)
        grad_y_plus = sign * _weno_plus(ph_y, y, lsm.ny)
        grad_z_minus = sign * _weno_minus(ph_z, z, lsm.nz)
        grad_z_plus = sign * _weno_plus(ph_z, z, lsm.nz)
        grad = 0.0
        grad_x_minus < 0 && (grad += grad_x_minus * grad_x_minus)
        grad_y_minus < 0 && (grad += grad_y_minus * grad_y_minus)
        grad_z_minus < 0 && (grad += grad_z_minus * grad_z_minus)
        grad_x_plus > 0 && (grad += grad_x_plus * grad_x_plus)
        grad_y_plus > 0 && (grad += grad_y_plus * grad_y_plus)
        grad_z_plus > 0 && (grad += grad_z_plus * grad_z_plus)
        grad = sqrt(grad)
        grad < 0.001 && (grad = 1.0)
        lsm.grid_gradient[i1] = grad
    end
    return lsm
end

# Advect the level set: phi = phi_temp + grid_vel * grid_gradient, then clamp
# the domain boundary to zero.
function advect!(lsm::LevelSet3D)
    compute_gradients!(lsm)
    for i in 1:(lsm.num_grid_pts)
        lsm.phi[i] = lsm.phi_temp[i] + lsm.grid_vel[i] * lsm.grid_gradient[i]
    end
    for i in 0:(lsm.nx), j in 0:(lsm.ny), k in 0:(lsm.nz)
        is_on_domain =
            i == 0 || i == lsm.nx || j == 0 || j == lsm.ny || k == 0 || k == lsm.nz
        idx = _idx3(lsm, i, j, k)
        if is_on_domain && lsm.phi[idx] > 0
            lsm.phi[idx] = 0.0
        end
    end
    return lsm
end

"""
    write_stl(level_set, filename; box_smooth=1)

Write the discretized zero surface of a [`LevelSet3D`](@ref) as an ASCII STL
file.
"""
function write_stl(lsm::LevelSet3D, filename::AbstractString; box_smooth::Integer=1)
    lsm.triangles = marching_cubes_3d(lsm.nx, lsm.ny, lsm.nz, lsm.phi)
    open(filename, "w") do io
        println(io, "solid mysolid")
        for tri in lsm.triangles
            println(io, "facet normal 0 0 0")
            println(io, "  outer loop")
            for p in tri
                println(io, "    vertex ", p[1], " ", p[2], " ", p[3], " ")
            end
            println(io, "  end loop")
            println(io, "endfacet")
        end
        return println(io, "endsolid mysolid")
    end
    return filename
end
