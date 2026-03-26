# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Development Commands

### Running Tests
```bash
# Run all tests
julia --project=. -e "using Pkg; Pkg.test()"

# Run specific test group (GROUP env var)
julia --project=. -e "ENV[\"GROUP\"]=\"Core_Tests\"; using Pkg; Pkg.test()"

# Available test groups: Core_Tests_1, Core_Tests_2, Examples_1, Examples_2, Examples_3, Examples_4, WCSMO14_1, WCSMO14_2
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
- `problem_types.jl`: Core problem types (`StiffnessTopOptProblem`, `HeatTransferTopOptProblem`, `HeatConductionProblem`, `PointLoadCantilever`, `HalfMBB`, `LBeam`, `TieBeam`, `RayProblem`, etc.)
- `grids.jl`: Grid/cell iteration utilities and problem metadata
- `matrices_and_vectors.jl`: Stiffness/conductivity matrix assembly and load/heat source vector construction
- `assemble.jl`: Ferrite-based finite element assembly
- `multiload.jl`: Multi-load case problems with `MultiLoad`
- `buckling.jl`: Buckling analysis support
- `elementinfo.jl`: Element FEA information (`ElementFEAInfo`, `GlobalFEAInfo`)
- `IO/`: INP file (Abaqus format) parser and VTK output

**TrussTopOptProblems** (`src/TrussTopOptProblems/`): Defines truss topology optimization problems
- `problem_types.jl`: `TrussProblem`, `PointLoadCantileverTruss` with truss-specific FEA
- `grids.jl`: `TrussGrid` for truss connectivity
- `matrices_and_vectors.jl`: Truss element stiffness matrices
- `elementinfo.jl`: Truss element information
- `TrussIO/`: Geo and JSON file parsers for truss problems (`load_truss_geo`, `load_truss_json`)

**Heat Transfer Problems**: The package supports heat transfer topology optimization
- `HeatConductionProblem`: 2D/3D steady-state heat conduction with surface heat flux (Neumann BC)
- `ThermalCompliance`: Objective function for thermal compliance (analogous to structural compliance)
- Key difference from structural: Heat flux is a surface load (NOT penalized), while body forces in structural mechanics ARE penalized because they depend on material density

**FEA** (`src/FEA/`): Finite element analysis solvers
- `solvers_api.jl`: Unified solver architecture with physics-based dispatch
- Key types: `GenericFEASolver{T,Physics,Solver}` - unified solver for all physics types
- Physics types: `LinearElasticity` (structural), `HeatTransfer` (thermal)
- Solver algorithms: `DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`
- `simulate.jl`: Simulation API with `simulate()` function
- `convergence_criteria.jl`: Convergence criteria (`DefaultCriteria`, `EnergyCriteria`)

**Functions** (`src/Functions/`): Differentiable objective/constraint functions
- `compliance.jl`: `Compliance` (strain energy) function for structural problems
- `thermal_compliance.jl`: `ThermalCompliance` and mean temperature functions for heat transfer
- `volume.jl`: `Volume` fraction function
- `stress_tensor.jl`: `StressTensor`, `ElementStressTensor` and von Mises stress
- `displacement.jl`: `Displacement` at specific DOFs
- `mean_compliance.jl`: `MeanCompliance` for uncertainty quantification
- `block_compliance.jl`: `BlockCompliance` for block-wise compliance
- `truss_stress.jl`: `TrussStress` for truss-specific stress functions
- `neural.jl`: Neural network surrogate models (`NeuralNetwork`, `TrainFunction`, `PredictFunction`, `NNParams`)
- `interpolation.jl`: `MaterialInterpolation`, `MultiMaterialVariables`, `element_densities`
- `trace.jl`: TopOpt trace functionality
- All functions extend `AbstractFunction{T}` from Nonconvex.jl and support automatic differentiation via Zygote

**CheqFilters** (`src/CheqFilters/`): Checkerboard and mesh-independent filtering
- `sens_filter.jl`: `SensFilter` (sensitivity filtering)
- `density_filter.jl`: `DensityFilter`, `ProjectedDensityFilter` (density filtering with optional projection)
- Filters implement `AbstractCheqFilter`, `AbstractSensFilter`, `AbstractDensityFilter` and are applied during optimization

**Algorithms** (`src/Algorithms/`): Topology optimization algorithms
- `beso.jl`: `BESO` (Bi-directional Evolutionary Structural Optimization)
- `geso.jl`: `GESO` (Genetic Evolutionary Structural Optimization)
- Note: `SIMP` and `ContinuationSIMP` are defined in `Functions/` using Nonconvex.jl optimization framework

**Utilities** (`src/Utilities/`): Shared utilities
- `penalties.jl`: Material interpolation penalties (`PowerPenalty`, `RationalPenalty`, `SinhPenalty`, `HeavisideProjection`, `SigmoidProjection`, `ProjectedPenalty`)
- `utils.jl`: Helper functions, macros (`@params`, `@forward_property`), and `RaggedArray`

### Key Design Patterns

**PseudoDensities**: A type-tracked array type (`src/TopOpt.jl:17-46`) that tracks whether densities have been interpolated (I), penalized (P), and/or filtered (F) at compile time via type parameters.

**Physics-Based Function Validation**: Functions in the `Functions` module assert the correct problem type:
- `Compliance`, `Displacement`, `StressTensor`: Require `StiffnessTopOptProblem` (structural mechanics)
- `ThermalCompliance`: Requires `HeatTransferTopOptProblem` (heat transfer)
- `TrussStress`: Requires `TrussProblem` (truss structures)

**Load Vector Assembly Architecture**: The FEA assembly distinguishes between penalized and non-penalized loads:
- `weights`/`fes`: Penalized by density (e.g., body forces in structural mechanics depend on material density)
- `dloads`/`fixedload`/`cload`: NOT penalized (e.g., surface traction, point forces, heat flux)
- For heat transfer: conductivity matrix is penalized, but heat flux (surface load) is NOT penalized
- This ensures correct physics: external loads are independent of material density distribution

**FEA Solver Abstraction**: All FEA solvers extend `AbstractFEASolver` and provide a common interface. The unified `GenericFEASolver{T,Physics,Solver}` uses two-layered dispatch:
- Physics layer (`LinearElasticity`, `HeatTransfer`): Determines element matrices and assembly
- Solver layer (`DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`): Determines linear system solution algorithm
- Physics type is automatically inferred from problem type via `physics_type(problem)` trait
- Use `FEASolver(DirectSolver, problem)` or `FEASolver(CGAssemblySolver, problem)` to create solvers (physics inferred)

**Optimization Framework**: Uses Nonconvex.jl for defining and solving optimization problems. Objectives and constraints are `AbstractFunction` instances that can be composed arbitrarily. Includes MMA variants (`MMA87`, `MMA02`), `Optimizer`, `SIMP`, `ContinuationSIMP`, and continuation methods (`PowerContinuation`, `ExponentialContinuation`, `LogarithmicContinuation`, `CubicSplineContinuation`, `SigmoidContinuation`).

**Package Extensions**: Makie visualization is implemented as a package extension (`ext/TopOptMakieExt/`) loaded when Makie is imported as a weak dependency.

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

# Create a heat conduction problem with surface heat flux
nels = (60, 20)
sizes = (1.0, 1.0)
k = 1.0  # thermal conductivity
heatflux = Dict{String,Float64}("top" => 100.0)  # heat flux on top boundary (W/m²)

problem = HeatConductionProblem(
    Val{:Linear}, nels, sizes, k;
    Tleft=100.0, Tright=0.0, heatflux=heatflux
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

### Truss Problem
```julia
using TopOpt

# Load truss from JSON file
truss = load_truss_json("path/to/truss.json")

# Create truss problem
problem = TrussProblem(truss, E=1.0, ν=0.3)

# Create solver
solver = FEASolver(DirectSolver, problem)

# Solve
solver.vars .= 1.0
solver()

# Compute truss stress
stress_fn = TrussStress(solver)
stresses = stress_fn(PseudoDensities(ones(length(solver.vars))))
```

## Dependencies

Core dependencies:
- **Ferrite** (v0.3.0): Finite element assembly (pinned version)
- **Nonconvex** (v2): Optimization framework with MMA, NLopt, Ipopt support
- **NonconvexMMA**: Method of Moving Asymptotes implementation
- **NonconvexPercival**: Percival solver for constrained optimization
- **NonconvexSemidefinite**: Semidefinite programming support
- **Zygote**: Reverse-mode automatic differentiation
- **ForwardDiff**: Forward-mode automatic differentiation
- **AbstractDifferentiation**: Unified AD interface
- **StaticArrays**: Performance-critical small arrays in FEA
- **Flux**: Neural network support
- **IterativeSolvers**: Conjugate gradient and other iterative methods
- **Preconditioners**: Preconditioning for iterative solvers
- **Ferrite**: Finite element assembly
- **VTKDataTypes**, **WriteVTK**: VTK output for visualization

Visualization (optional weak dependency):
- **Makie** (v0.23): Interactive visualization of results
- **GLMakie/CairoMakie**: Backends for Makie

## Key Exports

Main module exports from `TopOpt`:
- Core types: `PseudoDensities`, `TopOpt`
- Solvers: `FEASolver`, `DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`
- Optimization: `Optimizer`, `SIMP`, `ContinuationSIMP`, `BESO`, `GESO`
- Filters: `SensFilter`, `DensityFilter`
- Functions: `Compliance`, `ThermalCompliance`, `Displacement`, `Volume`, `TrussStress`, `MeanCompliance`, `BlockCompliance`
- Continuation: `PowerContinuation`, `ExponentialContinuation`, `LogarithmicContinuation`, `CubicSplineContinuation`, `SigmoidContinuation`, `Continuation`
- MMA variants: `MMA87`, `MMA02`
- Penalties: `PowerPenalty`, `RationalPenalty`, `SinhPenalty`, `HeavisideProjection`, `SigmoidProjection`, `ProjectedPenalty`
- Criteria: `DefaultCriteria`, `EnergyCriteria`
- IO: `save_mesh`
- Visualization: `visualize`