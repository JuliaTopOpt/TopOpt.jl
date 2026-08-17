// Reference driver for the level-set compliance-minimization comparison test.
//
// Reproduces the OpenLSTO compliance problem on a 40x20 cantilever with a 3x3
// grid of radius-2 holes, and prints `iteration compliance area` (full
// precision) to stdout so the Julia port in `test_compliance_reference.jl` can
// compare against it. Compile against a checkout of M2DOLab/OpenLSTO (see
// `.github/workflows/openlsto.yml`); the vendored Eigen must be replaced with
// 3.4.0 so the source builds with a modern GCC.

#include "M2DO_FEA.h"
#include "M2DO_LSM.h"

#include <cstdio>
#include <cstdlib>

using namespace std;
namespace FEA = M2DO_FEA;
namespace LSM = M2DO_LSM;

int main(int argc, char** argv) {
    const int max_iterations = argc > 1 ? atoi(argv[1]) : 40;
    const unsigned int nelx = 40, nely = 20;

    // Finite element setup (left edge fixed, downward point load at the
    // midpoint of the right edge).
    FEA::Mesh fea_mesh(2);
    MatrixXd fea_box(4, 2);
    fea_box << 0.0, 0.0, nelx, 0.0, nelx, nely, 0.0, nely;
    fea_mesh.MeshSolidHyperRectangle({int(nelx), int(nely)}, fea_box, 2, false);
    fea_mesh.is_structured = true;
    fea_mesh.AssignDof();

    fea_mesh.solid_materials.push_back(FEA::SolidMaterial(2, 1.0, 0.3, 1.0));
    FEA::StationaryStudy fea_study(fea_mesh);

    vector<double> coord = {0.0, 0.0}, tol = {1e-12, 1e10};
    vector<int> fixed_nodes = fea_mesh.GetNodesByCoordinates(coord, tol);
    vector<int> fixed_dof = fea_mesh.dof(fixed_nodes);
    vector<double> amplitude(fixed_dof.size(), 0.0);
    fea_study.AddBoundaryConditions(
        FEA::DirichletBoundaryConditions(fixed_dof, amplitude, fea_mesh.n_dof));

    coord = {1.0 * nelx, 0.5 * nely}, tol = {1e-12, 1e-12};
    vector<int> load_node = fea_mesh.GetNodesByCoordinates(coord, tol);
    vector<int> load_dof = fea_mesh.dof(load_node);
    vector<double> load_val(load_node.size() * 2);
    for (size_t i = 0; i < load_node.size(); ++i) {
        load_val[2 * i] = 0.0;
        load_val[2 * i + 1] = -0.5;
    }
    FEA::PointValues point_load(load_dof, load_val);
    fea_study.AssembleF(point_load, false);

    FEA::SensitivityAnalysis sens(fea_study);

    double move_limit = 0.5, band_width = 6;
    bool is_fixed_domain = false;

    vector<LSM::Hole> holes;
    holes.push_back(LSM::Hole(8, 4, 2));
    holes.push_back(LSM::Hole(16, 4, 2));
    holes.push_back(LSM::Hole(24, 4, 2));
    holes.push_back(LSM::Hole(12, 8, 2));
    holes.push_back(LSM::Hole(20, 8, 2));
    holes.push_back(LSM::Hole(28, 8, 2));
    holes.push_back(LSM::Hole(8, 12, 2));
    holes.push_back(LSM::Hole(16, 12, 2));
    holes.push_back(LSM::Hole(24, 12, 2));

    double max_area = 0.5, max_diff = 0.0001;
    vector<double> lambdas(2);

    LSM::Mesh lsm_mesh(nelx, nely, false);
    double mesh_area = lsm_mesh.width * lsm_mesh.height;
    LSM::LevelSet level_set(lsm_mesh, holes, move_limit, band_width, is_fixed_domain);
    level_set.reinitialise();
    LSM::Boundary boundary(level_set);
    LSM::MersenneTwister rng;

    unsigned int n_reinit = 0;
    int n_iterations = 0;
    vector<double> objective_values;
    double relative_difference = 1.0;

    while (n_iterations < max_iterations) {
        ++n_iterations;

        boundary.discretise(false, lambdas.size());
        boundary.computeAreaFractions();
        for (size_t i = 0; i < fea_mesh.solid_elements.size(); i++)
            fea_mesh.solid_elements[i].area_fraction =
                (lsm_mesh.elements[i].area < 1e-3) ? 1e-3 : lsm_mesh.elements[i].area;

        fea_study.AssembleKWithAreaFractions(false);
        fea_study.SolveWithCG();
        sens.ComputeComplianceSensitivities(false);

        for (size_t i = 0; i < boundary.points.size(); i++) {
            vector<double> boundary_point(2, 0.0);
            boundary_point[0] = boundary.points[i].coord.x;
            boundary_point[1] = boundary.points[i].coord.y;
            sens.ComputeBoundarySensitivities(boundary_point);
            boundary.points[i].sensitivities[0] = -sens.boundary_sensitivities[i];
            boundary.points[i].sensitivities[1] = -1;
        }
        sens.boundary_sensitivities.clear();

        double time_step;
        LSM::Optimise optimise(boundary.points, time_step, move_limit);
        optimise.length_x = lsm_mesh.width;
        optimise.length_y = lsm_mesh.height;
        optimise.boundary_area = boundary.area;
        optimise.mesh_area = mesh_area;
        optimise.max_area = max_area;
        optimise.Solve_With_NewtonRaphson();
        optimise.get_lambdas(lambdas);

        level_set.computeVelocities(boundary.points, time_step, 0, rng);
        level_set.computeGradients();
        bool is_reinitialised = level_set.update(time_step);

        if (!is_reinitialised) {
            if (n_reinit == 20) {
                level_set.reinitialise();
                n_reinit = 0;
            }
        } else
            n_reinit = 0;
        n_reinit++;

        double area = boundary.area / mesh_area;
        objective_values.push_back(sens.objective);

        if (n_iterations > 5) {
            double objective_value_k = sens.objective;
            relative_difference = 0.0;
            for (int i = 1; i <= 5; i++) {
                double objective_value_m = objective_values[n_iterations - i - 1];
                relative_difference =
                    max(relative_difference,
                        abs((objective_value_k - objective_value_m) / objective_value_k));
            }
        }

        printf("%d %.16e %.16e\n", n_iterations, sens.objective, area);
        if ((relative_difference < max_diff) & (area < 1.001 * max_area)) break;
    }

    return 0;
}
