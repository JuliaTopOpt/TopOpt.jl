# Architect mode rules

These rules apply when planning, designing, or strategizing before
implementation in this repository.

## Project scope

TopOpt.jl is a topology optimization framework for Julia, built on finite
element analysis (FEA) with automatic differentiation support. It targets
continuum and truss problems, heat transfer, and supports SIMP/BESO/GESO
optimization algorithms via the Nonconvex.jl ecosystem.

- **Language**: Julia (≥ 1.9). The LTS release is currently 1.10; prefer lower
  bounds in `[compat]` compatible with the LTS where possible.
- See `memory-bank/systemPatterns.md` for the full architecture reference:
  module structure, key design patterns, common problem API, dependencies, and
  usage examples. `memory-bank/` is git-ignored and kept up to date as the
  project evolves.

## Key architectural decisions

- **Modular design**: clear separation between problem definition
  (`TopOptProblems`, `TrussTopOptProblems`), FEA solvers (`FEA`), differentiable
  functions (`Functions`), filters (`CheqFilters`), and algorithms
  (`Algorithms`).
- **PseudoDensities**: a type-tracked array that tracks interpolation (I),
  penalization (P), and filtering (F) state via type parameters. Preserve this
  design in new code.
- **Physics-based dispatch**: `GenericFEASolver{T,Physics,Solver}` uses
  two-layered dispatch — physics (`LinearElasticity`, `HeatTransfer`) and
  solver (`DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`). Physics is
  inferred from the problem type via `physics_type(problem)`.
- **Load vector assembly**: penalized loads (`weights`/`fes`) vs non-penalized
  loads (`dloads`/`fixedload`/`cload`). External loads are independent of
  material density — this is a physics requirement, not a convention.
- **Nonconvex.jl integration**: objectives and constraints are
  `AbstractFunction` instances composable via Nonconvex. Includes MMA variants
  (`MMA87`, `MMA02`), `Optimizer`, `SIMP`, `ContinuationSIMP`, and continuation
  methods.
- **Package extensions**: Makie visualization is a weak-dependency extension
  (`ext/TopOptMakieExt/`), loaded only when Makie is imported.

## Design principles

- **Fail-fast** over silently trying to continue. Surface unexpected
  conditions so they can be inspected and understood.
- Use American spellings. Avoid jargon and metaphors not widely accepted by
  experts in the field. Do not make technical prose sound like a pitch deck.
- Favor **composability**: new functions should extend `AbstractFunction` and
  remain differentiable via Zygote. New solvers should extend
  `AbstractFEASolver` and follow the physics/solver dispatch pattern.
- Prefer **unconstrained type parameters** in `struct` constructors (see
  `.roo/rules-code/AGENTS.md` for the full cascade pattern). Do not over-constrain
  method signatures — annotate only as specifically as the implementation
  requires.
- When adding new packages to the local project, also update the `[compat]`
  section of `Project.toml` to bound the version of the new dependency. After
  editing `Project.toml`, run `Pkg.resolve()`.

## Memory Bank

This project uses a **Memory Bank** (`memory-bank/`, git-ignored) to preserve
context across agent sessions. The agent MUST read ALL memory bank files at
the start of EVERY task. When designing, update `systemPatterns.md` and
`activeContext.md` to reflect architectural decisions. See the root `AGENTS.md`
for the full Memory Bank structure.