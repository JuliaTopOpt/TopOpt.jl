# Circular hole used to seed the initial level-set (a port of
# `M2DO_LSM/include/hole.h`).

"""
    LevelSetHole(x, y, r)

A circular hole used to seed the initial level-set configuration: the region
inside the circle (distance to the centre less than `r`) is material-free.
"""
struct LevelSetHole
    coord::Coord
    r::Float64
end
function LevelSetHole(x::Real, y::Real, r::Real)
    return LevelSetHole(Coord(Float64(x), Float64(y)), Float64(r))
end

# Default "Swiss cheese" arrangement of holes (`LevelSet::initialise` in
# OpenLSTO): two interleaved grids of radius-5 circles with a 30-unit spacing.
function swiss_cheese_holes(width::Integer, height::Integer)
    w = Int(width)
    h = Int(height)
    # `std::round` in the C++ rounds half away from zero; Julia's default
    # `round(Int, x)` rounds half to even, which seeds one fewer hole row or
    # column on meshes whose width or height is 15 mod 60 (e.g. 75, 135, 195)
    # and rejects 75-wide meshes entirely.
    nx = round(Int, w / 30, RoundNearestTiesAway)
    ny = round(Int, h / 30, RoundNearestTiesAway)
    nx > 2 && ny > 2 || error("LevelSetMesh is too small for Swiss cheese initialisation.")
    n1 = nx * ny
    n2 = (nx - 1) * (ny - 1)
    holes = Vector{LevelSetHole}(undef, n1 + n2)
    dx = w / (2 * nx)
    dy = h / (2 * ny)

    for i in 1:n1
        x = (i - 1) % nx
        y = (i - 1) ÷ nx
        holes[i] = LevelSetHole(dx + 2 * x * dx, dy + 2 * y * dy, 5.0)
    end
    for i in 1:n2
        x = (i - 1) % (nx - 1)
        y = (i - 1) ÷ (nx - 1)
        holes[n1 + i] = LevelSetHole(2 * (dx + x * dx), 2 * (dy + y * dy), 5.0)
    end
    return holes
end
