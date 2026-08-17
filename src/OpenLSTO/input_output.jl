# Reading and writing of level-set data (a port of
# `M2DO_LSM/src/input_output.cpp`). The VTK files are ASCII ParaView formats
# (rectilinear grids for the signed distance and area fractions); the TXT
# files mirror OpenLSTO's plain-text dumps.

# Zero-padded 4-digit datapoint label, as in OpenLSTO's filename convention.
_zero_pad(datapoint::Integer) = lpad(string(datapoint), 4, '0')

function _vtk_output_path(
    datapoint::Integer, base::AbstractString, directory::AbstractString
)
    name = "$(base)_$(_zero_pad(datapoint)).vtk"
    return isempty(directory) ? name : joinpath(directory, name)
end

function _txt_output_path(
    datapoint::Integer, base::AbstractString, directory::AbstractString
)
    name = "$(base)_$(_zero_pad(datapoint)).txt"
    return isempty(directory) ? name : joinpath(directory, name)
end

# Rectilinear-grid VTK preamble shared by the level-set and area-fraction
# writers: a (width + 1) x (height + 1) grid of unit cells in the z = 0 plane.
function _write_rectilinear_grid_header(io::IO, mesh::LevelSetMesh)
    w = mesh.width
    h = mesh.height
    println(io, "# vtk DataFile Version 3.0")
    println(io, "Para0")
    println(io, "ASCII")
    println(io, "DATASET RECTILINEAR_GRID")
    println(io, "DIMENSIONS $(w + 1) $(h + 1) 1")
    println(io, "X_COORDINATES $(w + 1) int")
    for i in 0:w
        print(io, i, " ")
    end
    println(io)
    println(io, "Y_COORDINATES $(h + 1) int")
    for i in 0:h
        print(io, i, " ")
    end
    println(io)
    println(io, "Z_COORDINATES 1 int")
    println(io, "0")
    println(io)
    return nothing
end

"""
    save_level_set_vtk(level_set, datapoint; is_velocity=false, is_gradient=false, output_directory="")
    save_level_set_vtk(level_set, filename; is_velocity=false, is_gradient=false)

Write the signed distance function of a [`LevelSet`](@ref) as a ParaView VTK
rectilinear grid, optionally including the nodal velocity and gradient fields.
"""
function save_level_set_vtk(
    level_set::LevelSet,
    datapoint::Integer;
    is_velocity=false,
    is_gradient=false,
    output_directory="",
)
    return save_level_set_vtk(
        level_set,
        _vtk_output_path(datapoint, "level-set", output_directory);
        is_velocity,
        is_gradient,
    )
end

function save_level_set_vtk(
    level_set::LevelSet, filename::AbstractString; is_velocity=false, is_gradient=false
)
    open(filename, "w") do io
        _write_rectilinear_grid_header(io, level_set.mesh)
        println(io, "POINT_DATA $(level_set.mesh.nNodes)")
        println(io, "SCALARS distance float 1")
        println(io, "LOOKUP_TABLE default")
        for value in level_set.signedDistance
            println(io, value)
        end
        if is_velocity
            println(io, "SCALARS velocity float 1")
            println(io, "LOOKUP_TABLE default")
            for value in level_set.velocity
                println(io, value)
            end
        end
        if is_gradient
            println(io, "SCALARS gradient float 1")
            println(io, "LOOKUP_TABLE default")
            for value in level_set.gradient
                println(io, value)
            end
        end
    end
    return filename
end

"""
    save_level_set_txt(level_set, datapoint; output_directory="", is_xy=false)
    save_level_set_txt(level_set, filename; is_xy=false)

Write the signed distance, velocity, and gradient of a [`LevelSet`](@ref) as
plain text, optionally prefixed by the nodal x/y coordinates.
"""
function save_level_set_txt(
    level_set::LevelSet, datapoint::Integer; output_directory="", is_xy=false
)
    return save_level_set_txt(
        level_set, _txt_output_path(datapoint, "level-set", output_directory); is_xy
    )
end

function save_level_set_txt(level_set::LevelSet, filename::AbstractString; is_xy=false)
    open(filename, "w") do io
        for i in eachindex(level_set.signedDistance)
            if is_xy
                node = level_set.mesh.nodes[i]
                print(io, node.coord.x, " ", node.coord.y, " ")
            end
            println(
                io,
                level_set.signedDistance[i],
                " ",
                level_set.velocity[i],
                " ",
                level_set.gradient[i],
            )
        end
    end
    return filename
end

"""
    save_boundary_points_txt(boundary, datapoint; output_directory="")
    save_boundary_points_txt(boundary, filename)

Write the boundary points of a [`LevelSetBoundary`](@ref) (x, y, length) as
plain text.
"""
function save_boundary_points_txt(
    boundary::LevelSetBoundary, datapoint::Integer; output_directory=""
)
    return save_boundary_points_txt(
        boundary, _txt_output_path(datapoint, "boundary-points", output_directory)
    )
end

function save_boundary_points_txt(boundary::LevelSetBoundary, filename::AbstractString)
    open(filename, "w") do io
        for point in boundary.points
            println(io, point.coord.x, " ", point.coord.y, " ", point.length)
        end
    end
    return filename
end

"""
    save_boundary_segments_txt(boundary, datapoint; output_directory="")
    save_boundary_segments_txt(boundary, filename)

Write the boundary segments of a [`LevelSetBoundary`](@ref) (each as two
coordinate pairs separated by a blank line) as plain text.
"""
function save_boundary_segments_txt(
    boundary::LevelSetBoundary, datapoint::Integer; output_directory=""
)
    return save_boundary_segments_txt(
        boundary, _txt_output_path(datapoint, "boundary-segments", output_directory)
    )
end

function save_boundary_segments_txt(boundary::LevelSetBoundary, filename::AbstractString)
    open(filename, "w") do io
        for segment in boundary.segments
            start = boundary.points[segment.start]
            stop = boundary.points[segment.stop]
            println(io, start.coord.x, " ", start.coord.y)
            println(io, stop.coord.x, " ", stop.coord.y)
            println(io)
        end
    end
    return filename
end

"""
    save_area_fractions_vtk(mesh, datapoint; output_directory="")
    save_area_fractions_vtk(mesh, filename)

Write the element area fractions of a [`LevelSetMesh`](@ref) as a ParaView VTK
rectilinear grid (cell data).
"""
function save_area_fractions_vtk(
    mesh::LevelSetMesh, datapoint::Integer; output_directory=""
)
    return save_area_fractions_vtk(
        mesh, _vtk_output_path(datapoint, "area", output_directory)
    )
end

function save_area_fractions_vtk(mesh::LevelSetMesh, filename::AbstractString)
    open(filename, "w") do io
        _write_rectilinear_grid_header(io, mesh)
        println(io, "CELL_DATA $(mesh.nElements)")
        println(io, "SCALARS area float 1")
        println(io, "LOOKUP_TABLE default")
        for element in mesh.elements
            println(io, element.area)
        end
    end
    return filename
end

"""
    save_area_fractions_txt(mesh, datapoint; output_directory="", is_xy=false)
    save_area_fractions_txt(mesh, filename; is_xy=false)

Write the element area fractions of a [`LevelSetMesh`](@ref) as plain text,
optionally prefixed by the element centre coordinates.
"""
function save_area_fractions_txt(
    mesh::LevelSetMesh, datapoint::Integer; output_directory="", is_xy=false
)
    return save_area_fractions_txt(
        mesh, _txt_output_path(datapoint, "area", output_directory); is_xy
    )
end

function save_area_fractions_txt(mesh::LevelSetMesh, filename::AbstractString; is_xy=false)
    open(filename, "w") do io
        for element in mesh.elements
            if is_xy
                print(io, element.coord.x, " ", element.coord.y, " ")
            end
            println(io, element.area)
        end
    end
    return filename
end

"""
    boundary_vtk(filename, boundary_points, sensitivities, neighbors)

Write boundary points and their sensitivity fields as a ParaView VTK
unstructured grid of line segments. `boundary_points` is a vector of
coordinates, `sensitivities` a vector of per-field point values, and
`neighbors` the (start, end) index pairs of the segments.
"""
function boundary_vtk(
    filename::AbstractString,
    boundary_points::AbstractVector{<:AbstractVector},
    sensitivities::AbstractVector{<:AbstractVector},
    neighbors::AbstractVector{<:AbstractVector{<:Integer}},
)
    npoints = length(boundary_points)
    nsegments = length(neighbors)
    open(filename, "w") do io
        println(io, "# vtk DataFile Version 3.0")
        println(io, "Para0")
        println(io, "ASCII")
        println(io, "DATASET UNSTRUCTURED_GRID")
        println(io, "POINTS\t$(npoints)\tdouble")
        dim = length(first(boundary_points))
        for i in 1:npoints
            for j in 1:dim
                print(io, boundary_points[i][j], "\t")
            end
            println(io, "0")
        end
        println(io, "CELLS\t$(nsegments)\t$(3 * nsegments)")
        for i in 1:nsegments
            println(io, "2\t", neighbors[i][1], "\t", neighbors[i][2])
        end
        println(io, "CELL_TYPES\t$(nsegments)")
        for _ in 1:nsegments
            println(io, "3")
        end
        println(io, "POINT_DATA\t$(npoints)")
        for i in eachindex(sensitivities)
            println(io, "SCALARS\tSensitivity$(i)\tdouble\t1")
            println(io, "LOOKUP_TABLE DEFAULT")
            for j in 1:npoints
                println(io, sensitivities[i][j])
            end
            println(io)
        end
    end
    return true
end

"""
    write_optimisation_history_txt(objectives, constraints)

Write the optimisation history (`Output/optimisation_history.txt`): the
objective and constraint values, one row per iteration.
"""
function write_optimisation_history_txt(
    objectives::AbstractVector{<:Real},
    constraints::AbstractVector{<:AbstractVector{<:Real}},
)
    path = joinpath("Output", "optimisation_history.txt")
    mkpath(dirname(path))
    open(path, "w") do io
        for i in eachindex(objectives)
            print(io, objectives[i], " \t ", constraints[1][i])
            for j in 2:length(constraints)
                print(io, constraints[j][i], "\t")
            end
            println(io)
        end
    end
    return path
end
