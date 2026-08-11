# AGENTS.md

This file provides general meta instructions for agents working with code in
this repository. Detailed project description (architecture, module structure,
build/test commands, dependencies, usage examples) lives in the **Memory Bank**
(`memory-bank/`, git-ignored) — read those files for project specifics and keep
them up to date as the project evolves.

## Topic-specific guidance

opencode loads the topic-specific guidance files listed in `opencode.jsonc`
under `instructions`. They extend the guidance here with rules tailored to
specific kinds of work:

- [`.opencode/instructions/code.md`](.opencode/instructions/code.md) — code
  comments discipline, Julia generic indexing, `@inbounds` policy, and
  TopOpt.jl-specific coding notes.
- [`.opencode/instructions/debug.md`](.opencode/instructions/debug.md) —
  fail-fast guards, Revise/MCP usage, systematic debugging approach, and
  graphical display during debugging.
- [`.opencode/instructions/architect.md`](.opencode/instructions/architect.md)
  — project scope, key architectural decisions, and design principles.
- [`.opencode/instructions/ask.md`](.opencode/instructions/ask.md) —
  explanation stance, how to answer, and key TopOpt.jl concepts.

## Stance

- Favor **fail-fast** over silently trying to continue. Surface unexpected
  conditions so they can be inspected and understood.
- Use American spellings. Avoid jargon and metaphors not widely accepted by
  experts in the field. Do not make technical prose sound like a pitch deck.

## Progress updates

Always provide progress updates during tasks. For each meaningful step, state:

- **What you're doing** — the action being taken.
- **Why you're doing it** — the reason or goal behind the action.
- **What you found/changed** — the result or outcome.
- **What you'll do next** — the follow-up step.

Do not silently perform multiple tool calls without a progress update unless the
action is trivial (e.g., reading a single file, a quick grep).

## Memory Bank

This project uses a **Memory Bank** to preserve context across agent sessions,
because an agent's memory resets completely between sessions. After each reset,
the agent relies ENTIRELY on the Memory Bank to understand the project and
continue work effectively. The agent MUST read ALL memory bank files at the
start of EVERY task — this is not optional.

### Memory Bank Structure

The Memory Bank lives in `memory-bank/` (git-ignored) and consists of core files
in Markdown, building on each other in a clear hierarchy:

1. `projectbrief.md` — Foundation document; core requirements and goals; source
   of truth for project scope.
2. `productContext.md` — Why this project exists; problems it solves; how it
   should work; user experience goals.
3. `activeContext.md` — Current work focus; recent changes; next steps; active
   decisions and considerations; important patterns and preferences; learnings.
4. `systemPatterns.md` — System architecture; key technical decisions; design
   patterns in use; component relationships; critical implementation paths.
5. `techContext.md` — Technologies used; development setup; technical
   constraints; dependencies; tool usage patterns.
6. `progress.md` — What works; what's left to build; current status; known
   issues; evolution of project decisions.

Create additional files/folders within `memory-bank/` when they help organize
complex feature documentation, integration specs, API docs, testing strategies,
or deployment procedures.

### Memory Bank Updates

Memory Bank updates occur when:
1. Discovering new project patterns.
2. After implementing significant changes.
3. When the user requests with **update memory bank** (MUST review ALL files).
4. When context needs clarification.

After every memory reset, the agent begins completely fresh. The Memory Bank
is the only link to previous work. It must be maintained with precision and
clarity, because the agent's effectiveness depends entirely on its accuracy.

## Julia development

- Use the local `Project.toml` environment (`--project=.`). Revise, TestEnv,
  Cthulhu, and other developer-oriented tools live in the global (fallback)
  environment.
- Do not bias decisions about packages based on what is already installed.
- When adding new packages to the local project, also update the `[compat]`
  section of `Project.toml` to bound the version of the new dependency. After
  editing `Project.toml`, run `Pkg.resolve()`. Resolver errors sometimes
  indicate package conflict; `Pkg.update()` can fix such errors.
- Find the source for session-loaded packages with `Pkg.pkgdir(M::Module)`. For
  packages not loaded into the session, check the active project's
  `Manifest.toml` for the path before searching the hard drive.

### Debugging Julia code

- Use `Revise` to amortize compilation cost. The MCP server runs
  `Revise.revise()` automatically before every eval, so edits to loaded
  packages are already applied when code runs; calling `Revise.revise()` yourself
  is redundant. Exceptions: non-Revisable packages (Revise itself and its
  dependencies), and one-shot measurement/benchmarking runs — in those cases
  run julia directly from the shell.
- Use `Pkg.test()` for a final run only when ready to submit a pull request.

### Graphical display (Makie, Gtk, Qt, …)

Decide **at session start**, before loading any plotting backend, whether the
work is interactive or headless. Switching later requires reloading the backend
and restarting the session discards accumulated state.

Check what's available first:

    get(ENV, "DISPLAY", "")   # ":0" => a real monitor is available (e.g. WSLg)

- **Interactive (default for analysis/exploration).** When a real display is
  present, do not override `DISPLAY`:

      using GLMakie; GLMakie.activate!()
      display(fig)

- **Headless (CI-like batch, final `Pkg.test()`, profiling, or no real
  display).** Render to files with a non-interactive backend (CairoMakie →
  PNG/SVG), or start a virtual display once and point the session at it before
  loading the backend:

      # Bash: Xvfb :99 -screen 0 1024x768x24 &
      ENV["DISPLAY"] = ":99"

  Fall back to `xvfb-run julia ...` via Bash only for final `Pkg.test()` runs.

## Julia style guide

- Avoid being unnecessarily restrictive about method arguments.
  `f(A::Matrix{Float64})` silently excludes sparse matrices, GPU arrays,
  `Float32`, dual numbers, and anything else that would work fine — the caller
  gets a confusing `MethodError` instead. Annotate only as specifically as the
  implementation requires: use `Matrix{Float64}` only when a `ccall` or similar
  demands a specific memory layout and element type; use `AbstractMatrix` when
  2-D structure matters; use `AbstractArray` when it does not; leave unannotated
  when the method works for any input. Annotate to control dispatch and resolve
  ambiguities, not to document intent.
- The same caution applies to parametric `struct` constructors. Write the inner
  constructor with unconstrained value arguments —
  `MyStruct{A,B}(a, b) where {A,B}`, not `(a::A, b::B)` — and let the field
  declarations and `new` do the coercion; constraining the arguments breaks
  calls like `MyStruct{Float64}(1, 0)`. Outer constructors should only compute
  type parameters and delegate inward, forming a cascade
  `MyStruct(args...)` → `MyStruct{A}(args...)` → `MyStruct{A,B}(args...)` so
  every call form coerces identically. Some `struct`s have trailing
  type-parameters that are primarily internal, conferring inferrability but not
  usually manipulated by users; the cascade should leap over these by calling
  the inner constructor directly,
  `MyStruct{A,B}(args...)` → `MyStruct{A,B,typeof(c),typeof(d)}(args...)`,
  where `c` and `d` have already been `convert`ed to types consistent with `A`
  and `B`.
- Avoid redundant keyword syntax: when a variable name matches the keyword
  argument name, use the short form `f(; max_iter)` instead of
  `f(; max_iter=max_iter)`. This applies at function call sites, `NamedTuple`
  construction, and similar contexts.
- `@test_throws SomeExceptionType expr` may be worth testing when
  `SomeExceptionType` provides meaning, but
  `@test_throws "message that clearly explains the problem to users" expr` is
  typically the more relevant target for testing. There are cases where it may
  be reasonable to test both.

## Devops

- **AI-assistance disclosure (always required).** Every public-facing message
  produced by an agent — PR body, PR comment, GitHub issue, review, or any
  other visible post — **must** include a disclosure naming the model and
  harness/tool, placed at the end of the message:

      *Prepared with assistance from <model> via <tool>.*

  Example: *Prepared with assistance from qwen3.6-plus via opencode.*
  Never wait to be asked; never omit this. Do not present AI-assisted text
  as the maintainer's own unaided writing.
- Do not post comments on GitHub without getting explicit approval for the
  exact text. GitHub is also a social media environment; do not represent the
  maintainer without consent.
- Comments, docstrings, and commit messages must stand on their own for a
  reader who has only the repository: state what *is* true about the code now,
  not its history, its motivation, or the plan it came from. Re-read the diff's
  comments before proposing a commit. Full guidance and examples:
  `.opencode/instructions/code.md` (Code comments).
- Commit subject lines should ideally be shorter than lines in the body (aim
  for ≤ 50, up to 72 OK) due to formatting on GitHub.
- Changes motivated by GitHub issues or PRs should include a comment with the
  corresponding issue number. Do not put the issue number in the subject line,
  as that can be confusing in conjunction with a merge-squash that inserts the
  PR# in the subject. If the commit fixes an issue, put "Fixes #xyz" or similar
  in the body of the commit message; that will trigger GitHub to auto-close the
  issue. If a commit closes multiple issues, you cannot provide ranges or
  comma-separated lists of numbers; use "Fixes #abc; fixes #def; ...".
- For commits written by agents, include an `Assisted-by:` trailer (not
  `Co-authored-by:`) with the model and tool details:

      Assisted-by: qwen3.6-plus via opencode