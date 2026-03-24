# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Running Tests
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test group (GROUP env var)
julia --project=. -e "ENV[\"GROUP\"]=\"Core_Tests\"; using Pkg; Pkg.test()"

# Available test groups: Core_Tests, Extended_Tests, Examples_1, Examples_2, Examples_3, WCSMO14_1, WCSMO14_2
```

### Code Formatting
```bash
# Format all Julia files using JuliaFormatter
julia --project=. -e "using JuliaFormatter; format(\".\")"
```

### Documentation
```bash
# Build documentation locally
cd docs && julia --project=. make.jl

# Serve docs with LiveServer (from docs directory)
julia --project=. -e "using LiveServer; serve(dir=\"build\")"
```

### Package Management
```bash
# Activate project and instantiate dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Add a dependency
julia --project=. -e "using Pkg; Pkg.add(\"PackageName\")"

# Update dependencies
julia --project=. -e "using Pkg; Pkg.update()"
```

## High-Level Architecture

TopOpt.jl is a topology optimization framework built on finite element analysis (FEA) with automatic differentiation support. The architecture follows a modular design with clear separation between problem definition, FEA solvers, optimization functions, and algorithms.

### Module Structure

**TopOptProblems** (`src/TopOptProblems/`): Defines continuum topology optimization problems
- `problem_types.jl`: Core problem types (`StiffnessTopOptProblem`, `HeatTransferTopOptProblem`, `PointLoadCantilever`, `HalfMBB`, `HeatConductionProblem`, etc.)
- `grids.jl`: Grid/cell iteration utilities and problem metadata
- `matrices_and_vectors.jl`: Stiffness/conductivity matrix assembly and load/heat source vector construction
- `assemble.jl`: Ferrite-based finite element assembly
- `IO/`: INP file (Abaqus format) parser

**Heat Transfer Problems**: The package now supports heat transfer topology optimization
- `HeatConductionProblem`: 2D/3D steady-state heat conduction with volumetric heat generation
- `ThermalCompliance`: Objective function for thermal compliance (analogous to structural compliance)
- `MeanTemperature`: Objective function for average temperature minimization

**TrussTopOptProblems** (`src/TrussTopOptProblems/`): Defines truss topology optimization problems
- `problem_types.jl`: `TrussProblem` with truss-specific FEA
- `grids.jl`: `TrussGrid` for truss connectivity
- `matrices_and_vectors.jl`: Truss element stiffness matrices

**FEA** (`src/FEA/`): Finite element analysis solvers
- `solvers_api.jl`: Unified solver architecture with physics-based dispatch
- Key types: `GenericFEASolver{T,Physics,Solver}` - unified solver for all physics types
- Physics types: `LinearElasticity` (structural), `HeatTransfer` (thermal)
- Solver algorithms: `DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`

**Functions** (`src/Functions/`): Differentiable objective/constraint functions
- `compliance.jl`: Compliance (strain energy) function for structural problems
- `thermal_compliance.jl`: Thermal compliance and mean temperature functions for heat transfer
- `volume.jl`: Volume fraction function
- `stress_tensor.jl`: Stress tensor and von Mises stress
- `displacement.jl`: Displacement at specific DOFs
- `mean_compliance.jl`: Mean compliance for uncertainty quantification
- `truss_stress.jl`: Truss-specific stress functions
- `neural.jl`: Neural network surrogate models
- All functions extend `AbstractFunction{T}` from Nonconvex.jl and support automatic differentiation via Zygote

**CheqFilters** (`src/CheqFilters/`): Checkerboard and mesh-independent filtering
- `sens_filter.jl`: Sensitivity filtering
- `density_filter.jl`: Density filtering with optional projection
- Filters implement `AbstractCheqFilter` and are applied during optimization

**Algorithms** (`src/Algorithms/`): Topology optimization algorithms
- `beso.jl`: Bi-directional Evolutionary Structural Optimization (BESO)
- `geso.jl`: Genetic Evolutionary Structural Optimization (GESO)
- Note: SIMP and ContinuationSIMP are defined in `Functions/` using Nonconvex.jl optimization framework

**Utilities** (`src/Utilities/`): Shared utilities
- `penalties.jl`: Material interpolation penalties (Power, RAMP, Sinh, projection)
- `utils.jl`: Helper functions and macros

### Key Design Patterns

**PseudoDensities**: A type-tracked array type (`src/TopOpt.jl:11-46`) that tracks whether densities have been interpolated (I), penalized (P), and/or filtered (F) at compile time via type parameters.

**Physics-Based Function Validation**: Functions in the `Functions` module assert the correct problem type:
- `Compliance`, `Displacement`, `StressTensor`: Require `StiffnessTopOptProblem` (structural mechanics)
- `ThermalCompliance`, `MeanTemperature`: Require `HeatTransferTopOptProblem` (heat transfer)
- `TrussStress`: Requires `TrussProblem` (truss structures)

**FEA Solver Abstraction**: All FEA solvers extend `AbstractFEASolver` and provide a common interface. The unified `GenericFEASolver{T,Physics,Solver}` uses two-layered dispatch:
- Physics layer (`LinearElasticity`, `HeatTransfer`): Determines element matrices and assembly
- Solver layer (`DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`): Determines linear system solution algorithm
- Physics type is automatically inferred from problem type via `physics_type(problem)` trait
- Use `FEASolver(DirectSolver, problem)` or `FEASolver(CGAssemblySolver, problem)` to create solvers (physics inferred)

**Optimization Framework**: Uses Nonconvex.jl for defining and solving optimization problems. Objectives and constraints are `AbstractFunction` instances that can be composed arbitrarily.

**Package Extensions**: Makie visualization is implemented as a package extension (`ext/TopOptMakieExt/`) loaded when Makie is imported.

### Common Problem API

All problem types support a common interface:
- `getdim(problem)`: Get spatial dimension
- `floattype(problem)`: Get element type (e.g., `Float64`)
- `getncells(problem)`: Get number of elements
- `nnodespercell(problem)`: Get nodes per element
- `getgeomorder(problem)`: Get geometry order (1 or 2)
- `ndofs(problem.ch.dh)`: Get total number of DOFs

## Usage Examples

### Heat Transfer Problem
```julia
using TopOpt

# Create a heat conduction problem
nels = (60, 20)
sizes = (1.0, 1.0)
k = 1.0  # thermal conductivity
heat_source = 1.0  # volumetric heat generation

problem = HeatConductionProblem(
    Val{:Linear}, nels, sizes, k, heat_source;
    Tleft=0.0, Tright=0.0
)

# Create solver (physics automatically inferred from problem type)
solver = FEASolver(DirectSolver, problem; xmin=0.001)

# Set uniform density and solve
solver.vars .= 1.0
solver()

# Compute thermal compliance
comp = ThermalCompliance(solver)
val = comp(PseudoDensities(ones(length(solver.vars))))
```

### Dependencies

Core dependencies:
- **Ferrite** (v0.3.0): Finite element assembly (pinned version)
- **Nonconvex** (v2): Optimization framework with MMA, NLopt, Ipopt support
- **Zygote**: Reverse-mode automatic differentiation
- **ForwardDiff**: Forward-mode automatic differentiation
- **StaticArrays**: Performance-critical small arrays in FEA

Visualization (optional):
- **Makie**: Interactive visualization of results
- **GLMakie/CairoMakie**: Backends for Makie
