# TopOpt.jl Repository Health & Improvement Recommendations

Generated: 2026-08-07

## 1. Executive Summary

`TopOpt.jl` is a mature, feature-rich topology optimization package. Recent commit activity has been healthy (100 commits since 2026-01-01, ~90% test coverage target reached in #225), but several structural and dependency issues are keeping the package from staying current with the Julia ecosystem. The dominant blocker is a hard pin on **Ferrite v0.3.0**, which cascades into outdated transitive dependencies and prevents modernization.

---

## 2. Dependency Analysis

### 2.1 Direct dependencies (Project.toml)
The package declares **37 direct dependencies** plus **Makie as a weakdep**.

### 2.2 Outdated direct dependencies (holding back upgrades)

| Package | Current bound | Latest available | Risk / note |
|---|---|---|---|
| **Ferrite** | `=0.3.0` (hard pin) | v1.6.0 | 🔴 **Critical blocker** — API changed significantly; prevents latest Ferrite features and all downstream upgrades |
| **Flux** | `0.11, 0.12, 0.13, 0.14` | v0.16.11 | Neural surrogate code may need adaptation |
| **ForwardDiff** | `<0.10.35, 0.10, 1` | v1.4.5 | Bound excludes current 1.x line; `ADNLPModels` is the upper bound chain |
| **JSON** | `0.21` | v1.7.0 | Minor, but v1 is stable |
| **JuliaFormatter** | `1` | v2.12.4 | Formatter action still uses v1 |
| **StatsFuns** | `0.9, 1` | v2.2.1 | v2 may be breaking for some distribution code |
| **TimerOutputs** | `0.5` | v1.1.0 | Usually easy upgrade |
| **Zygote** | `0.6` | v0.7.12 | AD stack modernisation blocker |

### 2.3 Transitive dependency chains that are stuck

- `NonconvexPercival` → `ADNLPModels v0.4.0` → pins `ForwardDiff`, `NLPModels`, `LinearOperators`, etc.
- `NonconvexMMA` → `Optim v1.13.3` (Optim 2.x exists).
- `Preconditioners` → `AlgebraicMultigrid v0.5.1` (v2 exists).
- `VTKDataTypes` → `Colors v0.12`, `ResumableFunctions v0.6` (old).
- `JuliaFormatter` → `CommonMark v0.8`, `DataStructures v0.18`.

### 2.4 Dependency-reduction opportunity
Issue #214 explicitly asks to *reduce the number of direct dependencies*. Candidates to move/convert:

| Package | Current role | Recommendation |
|---|---|---|
| `JuliaFormatter` | Code formatting | Move to `[extras]` / test-only, or document as a dev tool; not needed at runtime |
| `Revise` | Development workflow | Remove from direct deps; devs can add it to their global env |
| `ColorSchemes`, `GeometryTypes` | Visualization helpers | Move into the Makie extension (`ext/TopOptMakieExt/`) |
| `FileIO`, `JSON` | IO utilities | Keep, but consider if `JSON3`/`JSON` consolidation is possible |
| `Flux` | Neural surrogate | Consider a package extension for neural-network functionality |
| `AbstractDifferentiation` | AD abstraction | Evaluate if it can be replaced by `DifferentiationInterface` (modern SciML AD interface) |

---

## 3. GitHub Issues Review

Recent open issues (sorted by relevance to maintainability):

| # | Title | Priority |
|---|---|---|
| **#212** | Update to the latest Ferrite | 🔴 Critical |
| **#214** | Reduce number of direct dependencies | 🔴 High |
| **#203** | Fix broken block compliance tests | 🟡 Medium |
| **#217** | More correctness tests for physics-based functions | 🟡 Medium |
| **#215** | Implement BESO for different physics types | 🟡 Medium |
| **#201** | More heat transfer problems / INP support for heat | 🟢 Feature |
| **#202** | Temperature function | 🟢 Feature |
| **#205** | Improve visualize and mesh writing for heat problems | 🟢 Feature |
| **#226** | Document every feature of TopOpt.jl | 🟢 Docs |
| **#227** | Parallelize doc build in CI | 🟢 CI |
| **#209** | Explore Mooncake as an alternative to Zygote | 🔵 Research |
| **#211–210–207–206** | New physics: acoustics, reaction-diffusion, Stokes/Darcy, electrostatics, diffusion | 🔵 Research |

Closed issues of note:
- #213 "Increase test coverage" → closed by #225 (target >90%).
- #218 "Simplify handling of non-design domains" → recently closed.
- #192 "SciML compatibility" → closed; verify `Nonconvex` still plays nicely with SciML stack.

---

## 4. Configuration & CI Findings

### 4.1 `Project.toml`
- `Makie` is correctly declared as a **weak dependency** with a package extension (`TopOptMakieExt`). Good.
- `julia = "1.9"` is reasonable, but CI now tests on Julia `1` (currently 1.12). Consider bumping to `1.10` LTS as minimum.
- `Ferrite = "=0.3.0"` is the single biggest compatibility anchor.

### 4.2 `.github/workflows/CI.yml`
- Uses `actions/checkout@v6` — up to date.
- Uses `codecov/codecov-action@v5` — up to date.
- Tests on `ubuntu-latest` only; consider adding `macos-latest` and `windows-latest` if the package claims cross-platform support.
- The `Core_Tests_Opposite_Preference` job sets a `LocalPreferences.toml` preference. This is good for regression testing the two code paths.
- The docs job does not use `julia-actions/cache@v2`; adding it would speed up repeated builds.

### 4.3 `.github/workflows/formatter.yml`
- Uses `create-pull-request@v3` — **outdated**; v7 is current.
- Uses global `Pkg.add("JuliaFormatter")` — will pull latest v2, while the project compat is v1. This mismatch can create PRs that only the global formatter understands.
- Recommendation: pin formatter version in workflow or run `julia --project=. -e "using JuliaFormatter; format(\".\")"` so it matches `Project.toml`.

### 4.4 `.github/workflows/CompatHelper.yml`
- Standard setup; ensure `COMPATHELPER_PRIV` secret is still valid.

### 4.5 `.JuliaFormatter.toml`
- Only `style = "blue"`. Fine.

### 4.6 `.codecov.yml`
- Only `comment: false`. Consider adding threshold/flags for test groups.

### 4.7 `LocalPreferences.toml`
- Contains `[TopOpt] penalty_before_interpolation = true`. Document that this is the default and how to toggle it.

### 4.8 `deps/build.jl`
- Empty file. Consider removing if not used, or add a deprecation note.

---

## 5. Recommended Action Plan

### Phase A: Unblock ecosystem upgrades (highest ROI)

1. **Ferrite migration (#212)**
   - Audit all Ferrite API usage (`src/TopOptProblems/assemble.jl`, `elementmatrix.jl`, `grids.jl`, `IO/`).
   - Create a feature branch, remove the `=0.3.0` pin, fix deprecations, and run the full test matrix.
   - This is the root cause of most dependency holdbacks.

2. **Trim direct dependencies (#214)**
   - Remove `Revise` and `JuliaFormatter` from `[deps]` (dev/test only).
   - Move `ColorSchemes`, `GeometryTypes`, `FileIO` (if only used for viz) into the Makie extension.
   - Consider making `Flux`/neural functions a weakdep extension.

3. **Fix known broken tests (#203)**
   - Run `test/Functions/test_block_compliance.jl` in isolation and investigate failures.

### Phase B: Modernize tooling

4. **Update formatter workflow**
   - Bump `create-pull-request` to v7.
   - Use the project's `JuliaFormatter` version (or update `Project.toml` compat to v2 and run v2 everywhere).

5. **Improve CI**
   - Add caching to the docs job.
   - Consider running docs build on PRs (currently only on push to master/tags).
   - Add `windows-latest` / `macos-latest` smoke tests if feasible.

6. **Codecov configuration**
   - Add `target: 90%` and `threshold: 1%` to align with the recent coverage push.

### Phase C: Code quality & new capabilities

7. **AD modernization (#209)**
   - After Ferrite is updated, evaluate `Mooncake` or `DifferentiationInterface` to replace/augment `Zygote` + `AbstractDifferentiation`.
   - This may also address long precompilation times observed during this review.

8. **Documentation (#226, #163)**
   - Fix strict Documenter build (#163).
   - Add doc pages for heat transfer and truss workflows (already have examples in `docs/src/literate/`).

9. **New physics/features**
   - Pick one of the physics issues (#201, #202, #205, #206, #207, #210, #211) as a focused milestone.

---

## 6. Quick Wins (can be done today)

- Remove empty `deps/build.jl`.
- Add cache action to docs CI job.
- Bump `peter-evans/create-pull-request@v3` → `v7`.
- Update `README.md` to recommend `julia = 1.10` and mention the Makie extension pattern.
- Verify `LocalPreferences.toml` is mentioned in docs/CLAUDE.md.
- Run `JuliaFormatter` with the project-bound version and commit any drift.

---

## 7. Risks

- Ferrite v1.x API changes will require non-trivial rewrites in assembly/grid code.
- Removing `Revise`/`JuliaFormatter` from direct deps is safe but may surprise contributors who rely on them being installed by default.
- Upgrading `Flux` may break `src/Functions/neural.jl`; add tests before upgrading.
- Upgrading `Zygote`/`ForwardDiff` may stress the custom `ChainRulesCore` rules; validate gradients with `FiniteDifferences`.

---

## 8. Conclusion

`TopOpt.jl` is functionally in good shape, but it is accumulating **ecosystem debt** around Ferrite and a few over-broad direct dependencies. The highest-impact improvement is the Ferrite upgrade (#212), followed by dependency trimming (#214) and CI/tooling modernization. Addressing these three items will make the package easier to install, faster to precompile, and simpler to maintain going forward.