# Debugging guidance

Guidance for troubleshooting issues, investigating errors, or diagnosing
problems in this repository.

## Fail-fast first

- Favor **fail-fast** over silently trying to continue. Surface unexpected
  conditions so they can be inspected and understood.
- When adding a guard or assertion, throw a descriptive error that names the
  values involved (`DimensionMismatch`, `ArgumentError`, or a custom message).
  Avoid bare `error()` calls that give no context.
- Prefer `@test_throws "message that clearly explains the problem to users" expr`
  when writing regression tests for the bug being fixed.

## Debugging Julia code

- Use `Revise` to amortize compilation cost. The MCP server runs
  `Revise.revise()` automatically before every eval, so edits to loaded
  packages are already applied when code runs; calling `Revise.revise()` yourself
  is redundant.
- Exceptions where Revise is not appropriate:
  + Debugging/developing non-Revisable packages (Revise itself and its
    dependencies).
  + One-shot measurement/benchmarking runs (cold package-load timing,
    invalidation analysis). Loading Revise can perturb results.
  In those cases, run julia directly from the shell.
- Use `Pkg.test()` for a final run only when ready to submit a pull request.
  During debugging, run the specific test group:
  `julia --project=. -e "ENV[\"GROUP\"]=\"Core_Tests\"; using Pkg; Pkg.test()"`.
- Find the source for session-loaded packages with `Pkg.pkgdir(M::Module)`.
  For packages not loaded into the session, check the active project's
  `Manifest.toml` for the path before searching the hard drive.

## Systematic debugging approach

1. **Reproduce** the error in the smallest possible example. Reduce noise
   before hypothesizing.
2. **Read the stack trace** carefully — Julia stack traces name the file and
   line of each frame. Identify whether the failure is in TopOpt.jl, a
   dependency, or user code.
3. **Inspect types** with `typeof`, `eltype`, `axes`, `size`. Many Julia bugs
   are type/shape mismatches that produce silent wrong answers only under
   specific dispatch.
4. **Check AD correctness** with FiniteDifferences when a Zygote gradient looks
   wrong. TopOpt functions extend `AbstractFunction` and must differentiate
   correctly.
5. **Isolate** the failing component: problem definition, FEA solver, objective
   function, filter, or algorithm. Test each layer independently.
6. **Add a regression test** that captures the bug before declaring it fixed.

## Graphical display during debugging

Decide **at session start**, before loading any plotting backend, whether the
work is interactive or headless. Switching later requires reloading the
backend and restarting the session discards accumulated state.

Check what's available first:

    get(ENV, "DISPLAY", "")   # ":0" => a real monitor is available (e.g. WSLg)

- **Interactive (default).** When a real display is present, do not override
  `DISPLAY`:

      using GLMakie; GLMakie.activate!()
      display(fig)

- **Headless.** Render to files with CairoMakie (PNG/SVG), or start a virtual
  display once and point the session at it before loading the backend:

      # Bash: Xvfb :99 -screen 0 1024x768x24 &
      ENV["DISPLAY"] = ":99"

  Fall back to `xvfb-run julia ...` via Bash only for final `Pkg.test()` runs.