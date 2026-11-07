// Reference driver for the level-set stress-minimization comparison test.
//
// Reproduces the OpenLSTO L-beam stress problem (`projects/stress_min/lbeam.cpp`)
// on a scaled 40x40 mesh with the five radius-4 holes, and prints
// `iteration objective tvm_max area` (full precision) to stdout so the Julia
// port in `test_stress_reference.jl` can compare against it. Compile against a
// checkout of M2DOLab/OpenLSTO (see `.github/workflows/openlsto.yml`); the
// vendored Eigen must be replaced with 3.4.0 so the source builds with a
// modern GCC.
//
// The 40x40 problem is the 100x100 `lbeam.cpp` setup scaled by 2/5: the inner
// corner is at x = y = 2/5*nelx = 16, the holes are the same five holes scaled
// to radius 4, and the tip load is applied at (40, 16) over the two nodes
// (39, 16) and (40, 16) with -1.5 each (total -3.0), matching the Julia port's
// `stress_minimization` defaults.

#include "M2DO_FEA.h"
#include "M2DO_LSM.h"

#include <cstdio>
#include <cstdlib>

using namespace std;
namespace FEA = M2DO_FEA;
namespace LSM = M2DO_LSM;

typedef unsigned int uint;

int main(int argc, char** argv) {
    const int max_iterations = argc > 1 ? atoi(argv[1]) : 40;
    const uint num_elem_x = 40, num_elem_y = 40;

    // Finite element setup (top edge fixed, point load on the right edge at
    // 2/5 height).
    FEA::Mesh fea_mesh(2);
    MatrixXd fea_box(4, 2);
    fea_box << 0, 0, num_elem_x, 0, num_elem_x, num_elem_y, 0, num_elem_y;
    vector<int> num_elem = {int(num_elem_x), int(num_elem_y)};
    fea_mesh.MeshSolidHyperRectangle(num_elem, fea_box, 2, false);
    fea_mesh.is_structured = true;
    fea_mesh.AssignDof();

    fea_mesh.solid_materials.push_back(FEA::SolidMaterial(2, 1.0, 0.3, 1.0));
    FEA::StationaryStudy fea_study(fea_mesh);

    vector<double> coord = {0.0, double(num_elem_y)};
    vector<double> tol = {double(num_elem_x) + 0.1, 0.1};
    vector<int> fixed_nodes = fea_mesh.GetNodesByCoordinates(coord, tol);
    vector<int> fixed_dof = fea_mesh.dof(fixed_nodes);
    vector<double> amplitude(fixed_dof.size(), 0.0);
    fea_study.AddBoundaryConditions(
        FEA::DirichletBoundaryConditions(fixed_dof, amplitude, fea_mesh.n_dof));

    vector<double> load_coord = {double(num_elem_x), double(num_elem_y) * 2.0 / 5.0};
    vector<double> load_tol = {1.1, 0.1};
    vector<int> load_nodes = fea_mesh.GetNodesByCoordinates(load_coord, load_tol);
    vector<int> load_dof = fea_mesh.dof(load_nodes);
    uint num_load_nodes = load_nodes.size();
    vector<double> load_values(load_nodes.size() * 2);
    for (uint i = 0; i < num_load_nodes; i++) {
        load_values[2 * i] = 0.0;
        load_values[2 * i + 1] = -3.0 / double(num_load_nodes);
    }
    FEA::PointValues point_load(load_dof, load_values);
    fea_study.AssembleF(point_load, false);

    int sens_type = 1;           // stress sensitivity
    double least_sq_radius = 2.0;
    FEA::SensitivityAnalysis sens(fea_study);

    // Level-set setup: L-beam domain with an inner corner at 2/5 width.
    double move_limit = 0.5, band_width = 6;
    bool is_fixed_domain = false, is_periodic = false;
    double num_reinit = 0;

    LSM::Mesh lsm_mesh(num_elem_x, num_elem_y, is_periodic);
    vector<LSM::Coord> vertical_edge(2), horizontal_edge(2);
    double inner_corner = double(num_elem_x) * 2 / 5;
    vertical_edge[0] = LSM::Coord({inner_corner - 0.01, inner_corner - 0.01});
    vertical_edge[1] = LSM::Coord({inner_corner + 0.01, num_elem_y + 0.01});
    horizontal_edge[0] = LSM::Coord({inner_corner - 0.01, inner_corner - 0.01});
    horizontal_edge[1] = LSM::Coord({num_elem_x + 0.01, inner_corner + 0.01});
    lsm_mesh.createMeshBoundary(vertical_edge);
    lsm_mesh.createMeshBoundary(horizontal_edge);

    vector<LSM::Hole> holes;
    double hole_radius = 4;
    holes.push_back(LSM::Hole(8, 8, hole_radius));
    holes.push_back(LSM::Hole(8, 20, hole_radius));
    holes.push_back(LSM::Hole(8, 32, hole_radius));
    holes.push_back(LSM::Hole(20, 8, hole_radius));
    holes.push_back(LSM::Hole(32, 8, hole_radius));

    LSM::LevelSet level_set(lsm_mesh, holes, move_limit, band_width, is_fixed_domain);
    vector<LSM::Coord> kill_region(2);
    kill_region[0] = LSM::Coord({inner_corner + 0.01, inner_corner + 0.01});
    kill_region[1] = LSM::Coord({num_elem_x + 0.01, num_elem_y + 0.01});
    level_set.killNodes(kill_region);
    level_set.createLevelSetBoundary(vertical_edge);
    level_set.createLevelSetBoundary(horizontal_edge);

    double tol_x = 3.01, tol_y = 2.01;
    double load_coord_x = num_elem_x, load_coord_y = double(num_elem_y) * 2 / 5;
    std::vector<LSM::Coord> points(2);
    points[0] = LSM::Coord({load_coord_x - tol_x, load_coord_y - tol_y});
    points[1] = LSM::Coord({load_coord_x + 0.01, load_coord_y + 0.01});
    level_set.fixNodes(points);
    points.clear();

    level_set.reinitialise();
    LSM::Boundary boundary(level_set);

    // Optimization parameters (mirroring lbeam.cpp).
    double max_area = 0.4;
    double mesh_area = lsm_mesh.width * lsm_mesh.height
        - pow(double(lsm_mesh.width) * 3 / 5, 2);
    double p_norm = 6;
    vector<double> objective_values;
    double relative_difference = 1.0;
    uint count_iter = 0;

    while (count_iter < max_iterations) {
        count_iter++;

        boundary.discretise(false, 2);
        boundary.computeAreaFractions();
        for (uint i = 0; i < fea_mesh.solid_elements.size(); i++) {
            if (lsm_mesh.elements[i].area < 1e-6)
                fea_mesh.solid_elements[i].area_fraction = 1e-6;
            else
                fea_mesh.solid_elements[i].area_fraction = lsm_mesh.elements[i].area;
        }

        fea_study.AssembleKWithAreaFractions(false);
        fea_study.AssembleF(point_load, false);
        fea_study.SolveWithCG();
        sens.ComputeStressSensitivities(false, p_norm);

        for (int i = 0; i < boundary.points.size(); i++) {
            vector<double> boundary_point(2, 0.0);
            boundary_point[0] = boundary.points[i].coord.x;
            boundary_point[1] = boundary.points[i].coord.y;
            sens.ComputeBoundarySensitivities(boundary_point, least_sq_radius, sens_type, p_norm);
            boundary.points[i].sensitivities[0] = -sens.boundary_sensitivities[i];
            boundary.points[i].sensitivities[1] = -1;
        }
        sens.boundary_sensitivities.clear();

        double time_step;
        vector<double> constraint_distances;
        constraint_distances.push_back(mesh_area * max_area - boundary.area);
        LSM::Optimise optimise(boundary.points, time_step, move_limit);
        optimise.length_x = lsm_mesh.width;
        optimise.length_y = lsm_mesh.height;
        optimise.boundary_area = boundary.area;
        optimise.mesh_area = mesh_area;
        optimise.max_area = max_area;
        double reduced_move_limit = 0.15;
        optimise.Solve_LbeamStress_With_NewtonRaphson(reduced_move_limit);

        level_set.computeVelocities(boundary.points);
        level_set.computeGradients();
        bool is_reinitialised = level_set.update(time_step);

        if (!is_reinitialised) {
            if (num_reinit == 1) {
                level_set.reinitialise();
                num_reinit = 0;
            }
        } else
            num_reinit = 0;
        num_reinit++;

        double area = boundary.area / mesh_area;
        objective_values.push_back(sens.objective);
        double objective_value_k, objective_value_m;
        if (count_iter > 5) {
            objective_value_k = sens.objective;
            relative_difference = 0.0;
            for (uint i = 1; i <= 5; i++) {
                objective_value_m = objective_values[count_iter - i - 1];
                relative_difference = max(
                    relative_difference,
                    abs((objective_value_k - objective_value_m) / objective_value_k));
            }
        }

        printf("%d %.16e %.16e %.16e\n", count_iter, sens.objective,
               sens.von_mises_max, area);
        if ((relative_difference <= 0.0005) & (area <= 1.001 * max_area)) break;
    }

    return 0;
}
