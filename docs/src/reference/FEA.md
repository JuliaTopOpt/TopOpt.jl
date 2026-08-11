# `FEA`

This sub-module of `TopOpt` defines the finite-element analysis solvers used
inside topology optimization. The solver dispatches on two orthogonal axes:
the **physics** (`LinearElasticity` or `HeatTransfer`) and the **linear-system
algorithm** (`DirectSolver`, `CGAssemblySolver`, or `CGMatrixFreeSolver`).

```@meta
CurrentModule = TopOpt.FEA
```

## Abstract types

```@docs
AbstractFEASolver
AbstractPhysics
LinearElasticity
HeatTransfer
AbstractLinearSolver
SolverResult
```

## Linear solvers

```@docs
DirectSolver
CGAssemblySolver
CGMatrixFreeSolver
```

## FEA solver

```@docs
GenericFEASolver
FEASolver
```

## Matrix-free operators

```@docs
MatrixFreeOperator
MatrixOperator
```

## Convergence criteria

```@docs
ConvergenceCriteria
DefaultCriteria
EnergyCriteria
```

## Forward simulation

```@docs
simulate
```