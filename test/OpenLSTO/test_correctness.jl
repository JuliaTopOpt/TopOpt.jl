# Unit tests for the OpenLSTO port-correctness fixes:
#  - `B_int` must accumulate `B * w * detJ` over the Gauss points (a uniform
#    strain field then gives the exact uniform stress).
#  - The Swiss-cheese seeding rounds half away from zero like `std::round`.
#  - The L-beam tip load is split over the two right-edge nodes nearest
#    (nelx, 2/5 nely), matching `lbeam.cpp`.
#  - The 3D Gauss rule is the 27-point (3x3x3) rule of `comp_min.cpp` and
#    integrates the hexahedron stiffness exactly.

using Test
using LinearAlgebra, SparseArrays, Random
using TopOpt

const L = TopOpt.OpenLSTO

@testset "OpenLSTO B_int uniform-strain stress" begin
    mesh = L.FEMesh(3, 3)
    material = L.SolidMaterial(1.0, 0.3, 1.0)
    study = L.StationaryStudy(mesh, material, Int[])
    sens = L.SensitivityAnalysis(study)

    # A uniform strain field u = (eps*x, eps*y) has strain [eps, 0, 0, eps]
    # in every element, so the B_int-based stress C * B_int * u_e must equal
    # the exact stress C * [eps, 0, 0, eps]. Without the Gauss-point
    # `w * detJ` factor in B_int the stress comes out 4x too large.
    eps = 0.001
    element = mesh.elements[1]
    ue = Float64[]
    for n in element.node_ids
        push!(ue, eps * mesh.coords[n, 1], eps * mesh.coords[n, 2])
    end
    stress = material.C * (sens.B_int * ue)
    expected = material.C * [eps, 0.0, 0.0, eps]
    @test stress ≈ expected rtol = 1e-10
end

@testset "OpenLSTO Swiss-cheese hole count" begin
    # `std::round` (C++) rounds half away from zero: w/30 = 2.5 -> 3 holes
    # across for a 75-wide mesh, and 4.5 -> 5 across for a 135-wide mesh.
    # Julia's default half-to-even rounding gives 2 and 4, and the 75-wide
    # mesh is then rejected by the `nx > 2` guard.
    holes75 = L.swiss_cheese_holes(75, 75)
    @test length(holes75) == 3 * 3 + 2 * 2
    holes135 = L.swiss_cheese_holes(135, 135)
    @test length(holes135) == 5 * 5 + 4 * 4
    # Non-tie widths are unaffected.
    holes90 = L.swiss_cheese_holes(90, 90)
    @test length(holes90) == 3 * 3 + 2 * 2
end

@testset "OpenLSTO L-beam two-node tip load" begin
    result = L.stress_minimization(; nelx=20, nely=20, max_iterations=1, verbose=false)
    bc = result.boundary_conditions
    @test length(bc.loads) == 2
    for (node, load) in bc.loads
        @test load ≈ [0.0, -1.5]
        # The two loaded nodes are (nelx - 1, 8) and (nelx, 8) on 20x20
        # (load_y = 2/5 * 20 = 8, 1-based node index = y * w + x + 1).
        @test node in (8 * 21 + 19 + 1, 8 * 21 + 20 + 1)
    end
    # The assembled load vector carries -1.5 on each loaded node's y-dof.
    @test sum(result.study.f) ≈ -3.0
    @test count(==(-1.5), result.study.f) == 2
end

@testset "OpenLSTO 3D Gauss rule" begin
    # The 27-point rule mirrors `GaussianQuadrature(spacedim=3, order=3)`:
    # points +-sqrt(3/5), 0 with weights 5/9, 8/9, 5/9 per direction.
    @test length(L._HEX_GAUSS_PTS) == 27
    @test length(L._HEX_GAUSS_WS) == 27
    @test sum(L._HEX_GAUSS_WS) ≈ 8.0                      # volume of [-1, 1]^3
    @test sum(w * p[1]^2 for (w, p) in zip(L._HEX_GAUSS_WS, L._HEX_GAUSS_PTS)) ≈ 8 / 3
    @test sum(w * p[1]^4 for (w, p) in zip(L._HEX_GAUSS_WS, L._HEX_GAUSS_PTS)) ≈ 8 / 5

    # Both the 27-point and the 2x2x2 rule integrate the trilinear-hex
    # stiffness integrand (quadratic per direction) exactly, so the element
    # stiffness must agree between the two rules.
    material = L.HexMaterial(1.0, 0.3, 1.0)
    Bs27 = [L._hex_B_matrix(p...) for p in L._HEX_GAUSS_PTS]
    ws27 = collect(L._HEX_GAUSS_WS)
    K27 = L._hex_element_stiffness(material, Bs27, ws27)
    pts8 = (
        (-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)),
        (1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)),
        (-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)),
        (1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)),
        (-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)),
        (1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)),
        (-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)),
        (1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)),
    )
    Bs8 = [L._hex_B_matrix(p...) for p in pts8]
    K8 = L._hex_element_stiffness(material, Bs8, ones(8))
    @test K27 ≈ K8 atol = 1e-13
end
