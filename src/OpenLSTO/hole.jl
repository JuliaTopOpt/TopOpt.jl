# Circular hole used to seed the initial level-set (a port of
# `M2DO_LSM/include/hole.h`).

struct Hole
    coord::Coord
    r::Float64
end
Hole(x::Real, y::Real, r::Real) = Hole(Coord(Float64(x), Float64(y)), Float64(r))

# Default "Swiss cheese" arrangement of holes (`LevelSet::initialise` in
# OpenLSTO): two interleaved grids of radius-5 circles with a 30-unit spacing.
function swiss_cheese_holes(width::Integer, height::Integer)
    w = Int(width)
    h = Int(height)
    nx = round(Int, w / 30)
    ny = round(Int, h / 30)
    nx > 2 && ny > 2 || error("Mesh is too small for Swiss cheese initialisation.")
    n1 = nx * ny
    n2 = (nx - 1) * (ny - 1)
    holes = Vector{Hole}(undef, n1 + n2)
    dx = w / (2 * nx)
    dy = h / (2 * ny)

    for i in 1:n1
        x = (i - 1) % nx
        y = (i - 1) ÷ nx
        holes[i] = Hole(dx + 2 * x * dx, dy + 2 * y * dy, 5.0)
    end
    for i in 1:n2
        x = (i - 1) % (nx - 1)
        y = (i - 1) ÷ (nx - 1)
        holes[n1 + i] = Hole(2 * (dx + x * dx), 2 * (dy + y * dy), 5.0)
    end
    return holes
end
