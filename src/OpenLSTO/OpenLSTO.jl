"""
    OpenLSTO

A Julia translation of the level-set topology optimization method in
OpenLSTO (https://github.com/M2DOLab/OpenLSTO), originally written in C++.

The module implements the 2D compliance-minimization workflow from
`projects/compliance/main.cpp` (with the hole-nucleation scheme from
`projects/hole_creation`), the 2D L-beam stress-minimization workflow from
`projects/stress_min/lbeam.cpp`, and the 3D compliance-minimization workflow
from `projects/3d/comp_min.cpp` (marching cubes, 3D fast marching, and a 3D
hexahedral solver), along with the Mersenne-Twister RNG and OpenLSTO's VTK/TXT
writers. A signed-distance level-set representation is advanced by boundary
velocities that satisfy the volume constraint, with the structural response
computed by an area-fraction ("ersatz material") finite element solve.

The translation is intentionally self-contained (its own mesh, fast marching,
boundary discretization, and finite element solver) so it can be verified
against the original before any of its pieces are replaced by the equivalent
machinery already present in the rest of TopOpt.jl.
"""
module OpenLSTO

using LinearAlgebra
using SparseArrays
using Random

export LevelSetMesh,
    LevelSetHole,
    LevelSet,
    LevelSetBoundary,
    FastMarchingMethod,
    Heap,
    LevelSetOptimizer,
    MersenneTwister,
    FEMesh,
    SolidMaterial,
    SolidElement,
    StationaryStudy,
    SensitivityAnalysis,
    compliance_minimization,
    stress_minimization,
    compliance_minimization_3d,
    LevelSetResult,
    LevelSetBoundaryConditions,
    area_fractions,
    LevelSet3D,
    HexMaterial,
    HexStudy,
    write_stl,
    save_level_set_vtk,
    save_level_set_txt,
    save_boundary_points_txt,
    save_boundary_segments_txt,
    save_area_fractions_vtk,
    save_area_fractions_txt,
    boundary_vtk,
    write_optimisation_history_txt

# Node and element status bit flags (mirror `NodeStatus` / `ElementStatus`
# in OpenLSTO's `M2DO_LSM/include/mesh.h`).
const NODE_NONE = 0
const NODE_INSIDE = 1
const NODE_OUTSIDE = 2
const NODE_BOUNDARY = 4
const NODE_CUT = NODE_INSIDE | NODE_OUTSIDE

const ELEMENT_NONE = 0
const ELEMENT_INSIDE = 1
const ELEMENT_OUTSIDE = 2
const ELEMENT_CENTRE_INSIDE = 4
const ELEMENT_CENTRE_OUTSIDE = 8

# Fast-marching node status flags (mirror `FMM_NodeStatus` in
# `M2DO_LSM/include/fast_marching_method.h`).
const FMM_NONE = 0
const FMM_FROZEN = 1
const FMM_TRIAL = 2
const FMM_MASKED = 4

include("common.jl")
include("mesh.jl")
include("hole.jl")
include("heap.jl")
include("mersenne_twister.jl")
include("fast_marching.jl")
include("level_set.jl")
include("boundary.jl")
include("input_output.jl")
include("hole_nucleation.jl")
include("fea.jl")
include("optimise.jl")
include("optimiser.jl")
include("stress_minimization.jl")
include("mc_table.jl")
include("marching_cubes.jl")
include("fea_3d.jl")
include("level_set_3d.jl")
include("compliance_minimization_3d.jl")

end # module
