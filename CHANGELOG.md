
## 0.2.0 (2021-02-09)

### Added
* Added `TrussTopOptProblems` module.
* Added `visualize` method for visualization FEA deformation and optimized results for both continuum and truss problems.
* Added `buckling` functions to `TopOptProblems`, but not well tested yet.

### Changed
* Wrapped all visualization methods into the `TopOptProblems.Visualization` module

## 0.1.0 (2021-02-08)

This is initial release of TopOpt.jl. There are a number of features available including:

1. Volume-constrained compliance minimisation for 2D and 3D unstructured meshes.
2. Density and sensitivity chequerboard filters.
3. Heaviside projection.
4. Method of moving asymptotes and the solid isotropic material with penalisation method.
5. Advanced continuation methods.
6. Bi-directional evolutionary structural optimisation for compliance minimisation.
7. Genetic evolutionary structural optimisation for compliance minimisation.
8. Direct and iterative linear system solvers.
9. Inp file input and vtk file output for end-to-end topology optimisation.

A number of experimental features are also there including:

1. Stress constrained optimisation and maximum stress minimisation
2. Robust maximum compliance constrained optimisation for multiple loading scenarios.
3. Stochastic and risk-averse compliance minimisation.