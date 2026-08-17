# `OpenLSTO`

This sub-module of `TopOpt` is a self-contained Julia translation of the
level-set topology optimization method in
[OpenLSTO](https://github.com/M2DOLab/OpenLSTO). It implements the 2D
compliance-minimization workflow (with optional hole nucleation) and the 2D
L-beam stress-minimization workflow, on its own mesh, fast marching, boundary
discretization, and finite element solver.

```@meta
CurrentModule = TopOpt.OpenLSTO
```

## Optimization drivers

```@docs
compliance_minimization
stress_minimization
compliance_minimization_3d
LevelSetResult
LevelSetBoundaryConditions
area_fractions
```

## Level-set representation

```@docs
LevelSetMesh
LevelSet
LevelSetHole
LevelSetBoundary
```

## Fast marching and optimization

```@docs
FastMarchingMethod
Heap
MersenneTwister
integer
normal
get_seed
set_seed!
LevelSetOptimizer
```

## Finite element analysis

```@docs
FEMesh
SolidMaterial
SolidElement
StationaryStudy
SensitivityAnalysis
HexMaterial
HexStudy
```

## 3D level-set method

```@docs
LevelSet3D
write_stl
```

## Internal helpers

```@docs
discretise!
compute_area_fractions!
march!
compute_compliance_sensitivities!
compute_stress_sensitivities!
compute_boundary_sensitivity
```

## Input and output

```@docs
save_level_set_vtk
save_level_set_txt
save_boundary_points_txt
save_boundary_segments_txt
save_area_fractions_vtk
save_area_fractions_txt
boundary_vtk
write_optimisation_history_txt
```
