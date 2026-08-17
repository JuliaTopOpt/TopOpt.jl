# Marching cubes (a port of `M2DO_3D_LSM/marching_cubes_cross.cpp`). The
# lookup tables live in `mc_table.jl`.

# Linear interpolation of the point where a triangle vertex crosses the edge
# between two grid points, at the given iso value.
function _linear_interp(p1::NTuple{4,Float64}, p2::NTuple{4,Float64}, value::Float64)
    if p1[4] != p2[4]
        t = (value - p1[4]) / (p2[4] - p1[4])
        return (
            p1[1] + (p2[1] - p1[1]) * t,
            p1[2] + (p2[2] - p1[2]) * t,
            p1[3] + (p2[3] - p1[3]) * t,
        )
    else
        return (p1[1], p1[2], p1[3])
    end
end

# The 12 edges of a cube, as pairs of vertex indices (0-based).
const _MC_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

"""
    marching_cubes_3d(nx, ny, nz, phi; iso_value=0.0)

Discretize the `iso_value` contour of a 3D signed-distance field `phi` (grid
points ordered z-fastest, then y, then x) into a list of triangles. Each
triangle is a tuple of three `(x, y, z)` vertices.
"""
function marching_cubes_3d(
    nx::Integer, ny::Integer, nz::Integer, phi::Vector{Float64}; iso_value::Float64=0.0
)
    nx = Int(nx)
    ny = Int(ny)
    nz = Int(nz)
    yz = (ny + 1) * (nz + 1)
    triangles = NTuple{3,NTuple{3,Float64}}[]

    for i in 0:(nx - 1)
        for j in 0:(ny - 1)
            for k in 0:(nz - 1)
                ind = k + j * (nz + 1) + i * yz
                verts = NTuple{4,Float64}[
                    (Float64(i), Float64(j), Float64(k), phi[ind + 1]),
                    (Float64(i + 1), Float64(j), Float64(k), phi[ind + yz + 1]),
                    (Float64(i + 1), Float64(j), Float64(k + 1), phi[ind + yz + 1 + 1]),
                    (Float64(i), Float64(j), Float64(k + 1), phi[ind + 1 + 1]),
                    (Float64(i), Float64(j + 1), Float64(k), phi[ind + (nz + 1) + 1]),
                    (
                        Float64(i + 1),
                        Float64(j + 1),
                        Float64(k),
                        phi[ind + yz + (nz + 1) + 1],
                    ),
                    (
                        Float64(i + 1),
                        Float64(j + 1),
                        Float64(k + 1),
                        phi[ind + yz + (nz + 1) + 1 + 1],
                    ),
                    (
                        Float64(i),
                        Float64(j + 1),
                        Float64(k + 1),
                        phi[ind + (nz + 1) + 1 + 1],
                    ),
                ]

                cube_index = 0
                for n in 1:8
                    if verts[n][4] <= iso_value
                        cube_index |= (1 << (n - 1))
                    end
                end

                edge_table = MC_EDGE_TABLE[cube_index + 1]
                edge_table == 0 && continue

                int_verts = Vector{NTuple{3,Float64}}(undef, 12)
                for e in 1:12
                    if edge_table & (1 << (e - 1)) != 0
                        a, b = _MC_EDGES[e]
                        int_verts[e] = _linear_interp(verts[a + 1], verts[b + 1], iso_value)
                    end
                end

                tri = MC_TRI_TABLE[cube_index + 1]
                n = 1
                while n <= 16 && tri[n] != -1
                    push!(
                        triangles,
                        (
                            int_verts[tri[n + 2] + 1],
                            int_verts[tri[n + 1] + 1],
                            int_verts[tri[n] + 1],
                        ),
                    )
                    n += 3
                end
            end
        end
    end
    return triangles
end
