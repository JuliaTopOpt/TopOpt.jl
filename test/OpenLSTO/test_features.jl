# Tests for the OpenLSTO port's auxiliary features: the Mersenne-Twister RNG,
# the VTK/TXT writers, hole nucleation, and the L-beam stress minimization.

using Test
using TopOpt

const L = TopOpt.OpenLSTO

@testset "OpenLSTO MersenneTwister" begin
    rng = L.MersenneTwister()
    @test 0.0 <= rng() <= 1.0
    @test 1 <= L.integer(rng, 1, 10) <= 10
    @test isfinite(L.normal(rng))

    L.set_seed!(rng, 1234)
    @test L.get_seed(rng) == UInt32(1234)
    a = rng()
    L.set_seed!(rng, 1234)
    @test rng() == a
end

@testset "OpenLSTO level-set accessors" begin
    mesh = L.LevelSetMesh(20, 10)
    @test mesh.width == 20 && mesh.height == 10
    @test mesh.nNodes == 21 * 11
    @test mesh.nElements == 200

    ls = L.LevelSet(mesh, L.LevelSetHole[L.LevelSetHole(10, 5, 3)], 0.5, 6, false)
    L.reinitialise!(ls)
    boundary = L.LevelSetBoundary(ls)
    L.discretise!(boundary, 2)
    area = L.compute_area_fractions!(boundary)
    @test 0.0 <= area <= 200.0
    @test all(0.0 .<= [e.area for e in mesh.elements] .<= 1.0)
end

@testset "OpenLSTO level-set editing" begin
    mesh = L.LevelSetMesh(20, 10)

    # create_mesh_boundary! marks nodes inside a rectangle as domain boundary.
    L.create_mesh_boundary!(mesh, L.Coord[L.Coord(5.0, 5.0), L.Coord(10.0, 10.0)])
    @test any(n.isDomain for n in mesh.nodes)

    ls = L.LevelSet(mesh, L.LevelSetHole[L.LevelSetHole(10, 5, 3)], 0.5, 6, false)
    L.reinitialise!(ls)

    # kill_nodes! zeroes out the signed distance and fixes the nodes in a region.
    L.kill_nodes!(ls, L.Coord[L.Coord(15.0, 5.0), L.Coord(20.0, 10.0)])
    @test any(n.isFixed for n in mesh.nodes)
    @test any(ls.signedDistance .== -1e-6)

    # fix_nodes! fixes nodes; create_level_set_boundary! zeroes the signed distance.
    L.create_level_set_boundary!(ls, L.Coord[L.Coord(0.0, 0.0), L.Coord(2.0, 2.0)])
    @test any(ls.signedDistance .== 0.0)
    L.fix_nodes!(ls, L.Coord[L.Coord(0.0, 0.0), L.Coord(2.0, 2.0)])
    @test any(n.isFixed for n in mesh.nodes)
end

@testset "OpenLSTO stochastic velocity extension" begin
    mesh = L.LevelSetMesh(20, 10)
    ls = L.LevelSet(mesh, L.LevelSetHole[L.LevelSetHole(10, 5, 3)], 0.5, 6, false)
    L.reinitialise!(ls)
    boundary = L.LevelSetBoundary(ls)
    L.discretise!(boundary, 2)

    rng = L.MersenneTwister()
    # Zero temperature adds no noise, so the time step is returned unchanged.
    @test L.compute_velocities!(ls, boundary.points, 1.0, 0.0, rng) == 1.0
end

@testset "OpenLSTO input/output" begin
    mktempdir() do dir
        holes = L.LevelSetHole[
            L.LevelSetHole(8, 4, 2),
            L.LevelSetHole(16, 4, 2),
            L.LevelSetHole(24, 4, 2),
            L.LevelSetHole(12, 8, 2),
            L.LevelSetHole(20, 8, 2),
            L.LevelSetHole(28, 8, 2),
            L.LevelSetHole(8, 12, 2),
            L.LevelSetHole(16, 12, 2),
            L.LevelSetHole(24, 12, 2),
        ]
        result = L.compliance_minimization(;
            nelx=40, nely=20, holes=holes, max_iterations=2, verbose=false
        )

        @test isfile(L.save_level_set_vtk(result.level_set, 1; output_directory=dir))
        @test isfile(
            L.save_area_fractions_vtk(result.level_set.mesh, 1; output_directory=dir)
        )
        @test isfile(L.save_boundary_segments_txt(result.boundary, 1; output_directory=dir))
        @test isfile(L.save_level_set_txt(result.level_set, 1; output_directory=dir))
        @test isfile(
            L.save_area_fractions_txt(result.level_set.mesh, 1; output_directory=dir)
        )
        @test isfile(L.save_boundary_points_txt(result.boundary, 1; output_directory=dir))

        vtk = read(L.save_level_set_vtk(result.level_set, 1; output_directory=dir), String)
        @test occursin("DATASET RECTILINEAR_GRID", vtk)
        @test occursin("DIMENSIONS 41 21 1", vtk)
    end
end

@testset "OpenLSTO hole nucleation" begin
    mesh = L.LevelSetMesh(20, 10)
    ls = L.LevelSet(mesh, L.LevelSetHole[L.LevelSetHole(10, 5, 3)], 0.5, 6, false)
    L.reinitialise!(ls)
    count, h_index, h_elem = L.hole_map(mesh, ls, 1.0, 2.0)
    @test count >= 0
    @test length(h_index) == mesh.nNodes
    @test length(h_elem) == mesh.nElements

    h_lsf = fill(1.0, mesh.nNodes)
    h_nsens = [ones(2) for _ in 1:(mesh.nNodes)]
    gammas = [0.1, 0.2]
    L.get_h_lsf!(h_index, h_nsens, gammas, h_lsf)
    @test all(isfinite, h_lsf)

    area = L.hole_area_fractions(mesh, ls.signedDistance)
    @test 0.0 <= area <= mesh.nElements

    holes = L.LevelSetHole[
        L.LevelSetHole(8, 4, 2),
        L.LevelSetHole(16, 4, 2),
        L.LevelSetHole(24, 4, 2),
        L.LevelSetHole(12, 8, 2),
        L.LevelSetHole(20, 8, 2),
        L.LevelSetHole(28, 8, 2),
        L.LevelSetHole(8, 12, 2),
        L.LevelSetHole(16, 12, 2),
        L.LevelSetHole(24, 12, 2),
    ]
    result = L.compliance_minimization(;
        nelx=40, nely=20, holes=holes, max_iterations=8, hole_nucleation=true, verbose=false
    )
    @test result isa L.LevelSetResult
    @test length(result.objectives) >= 1
    @test length(result.areas) == length(result.objectives)
end

@testset "OpenLSTO stress minimization" begin
    result = L.stress_minimization(; nelx=20, nely=20, max_iterations=4, verbose=false)
    @test result isa L.LevelSetResult
    @test length(result.objectives) == 4
    @test length(result.areas) == 4
    @test all(isfinite, result.objectives)
    @test result.sensitivities.von_mises_max > 0.0
end

@testset "OpenLSTO 3D compliance minimization" begin
    result = L.compliance_minimization_3d(;
        nelx=8, nely=4, nelz=4, max_iterations=3, verbose=false
    )
    @test length(result.compliances) == 3
    @test length(result.areas) == 3
    @test result.level_set isa L.LevelSet3D
    @test result.study isa L.HexStudy
    @test all(isfinite, result.compliances)
    @test all(0.0 .<= result.areas .<= 1.0)

    triangles = L.marching_cubes_3d(8, 4, 4, result.level_set.phi)
    @test !isempty(triangles)

    mktempdir() do dir
        stl = L.write_stl(result.level_set, joinpath(dir, "mystlfile.stl"))
        @test isfile(stl)
        @test occursin("solid mysolid", read(stl, String))
    end
end

@testset "OpenLSTO boundary VTK and history" begin
    mktempdir() do dir
        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        sens = [[0.1, 0.2, 0.3], [-1.0, -1.0, -1.0]]
        neighbors = [[0, 1], [1, 2]]
        path = joinpath(dir, "boundary.vtk")
        @test L.boundary_vtk(path, pts, sens, neighbors)
        content = read(path, String)
        @test occursin("DATASET UNSTRUCTURED_GRID", content)
        @test occursin("POINTS\t3\tdouble", content)
    end

    mktempdir() do dir
        cd(dir) do
            L.write_optimisation_history_txt([1.0, 2.0], [[0.5, 0.4]])
            @test isfile(joinpath("Output", "optimisation_history.txt"))
        end
    end
end
