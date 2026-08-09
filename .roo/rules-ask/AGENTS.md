# Ask mode rules

These rules apply when providing explanations, documentation, or answers to
technical questions about this repository.

## Stance

- Use American spellings. Avoid jargon and metaphors not widely accepted by
  experts in the field. Do not make technical prose sound like a pitch deck.
- Favor **fail-fast** over silently trying to continue — when explaining code,
  point out where it silently continues vs. where it surfaces errors.
- Be precise about what the code *does now*, not what it was intended to do or
  what it might do in the future. Cite the actual source.

## How to answer

- Ground explanations in the actual source. Reference files and line numbers
  when describing behavior.
- For architecture questions, consult `memory-bank/systemPatterns.md` (the
  full architecture reference) and the root `AGENTS.md` (project context and
  conventions).
- For Julia-specific questions (indexing, `@inbounds`, style), consult
  `.roo/rules-code/AGENTS.md`.
- For debugging questions, consult `.roo/rules-debug/AGENTS.md`.
- For design questions, consult `.roo/rules-architect/AGENTS.md`.

## Key concepts in TopOpt.jl

- **PseudoDensities**: a type-tracked array tracking interpolation (I),
  penalization (P), and filtering (F) state via type parameters
  (`src/TopOpt.jl`).
- **FEA solver abstraction**: `GenericFEASolver{T,Physics,Solver}` with
  physics-based dispatch (`LinearElasticity`, `HeatTransfer`) and solver
  dispatch (`DirectSolver`, `CGAssemblySolver`, `CGMatrixFreeSolver`).
- **Load vector assembly**: penalized loads (`weights`/`fes`) vs non-penalized
  loads (`dloads`/`fixedload`/`cload`). External loads are independent of
  material density.
- **Differentiable functions**: all objectives/constraints extend
  `AbstractFunction` from Nonconvex.jl and support AD via Zygote.
- **Optimization**: Nonconvex.jl framework with MMA variants, `SIMP`,
  `ContinuationSIMP`, and continuation methods.

## Julia development context

- Use the local `Project.toml` environment (`--project=.`).
- Find the source for session-loaded packages with `Pkg.pkgdir(M::Module)`.
  For packages not loaded into the session, check the active project's
  `Manifest.toml` for the path.
- Use `Revise` to amortize compilation cost. The MCP server runs
  `Revise.revise()` automatically before every eval.
- Use `Pkg.test()` for a final run only when ready to submit a pull request.
  Test groups are selected via the `GROUP` env var (`Core_Tests`,
  `Examples_1`–`Examples_4`, `WCSMO14_1`, `WCSMO14_2`).

## Memory Bank

This project uses a **Memory Bank** (`memory-bank/`, git-ignored) to preserve
context across agent sessions. The agent MUST read ALL memory bank files at
the start of EVERY task. See the root `AGENTS.md` for the full structure.