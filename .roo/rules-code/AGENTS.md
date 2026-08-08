# Code mode rules

These rules apply when writing, modifying, or refactoring code in this
repository. They adapt guidance from
[timholy/claude_config](https://github.com/timholy/claude_config) to TopOpt.jl.

## Code comments

A comment explains the code that is there *now*: an invariant it must maintain,
a non-obvious reason it has to be this way, a constraint a future editor would
otherwise break. Write every comment so it still reads correctly to someone who
has only the repository in front of them — no plan, no chat log, no memory of
how the code came to be. If a sentence only makes sense to someone who watched
the code being written, it does not belong in the source.

State what *is* true, not what *was* true or *why you happened to write it
today*:

- ✗ `# CHUNK-3: reuse the sparse format chosen earlier so alignment stays fast`
- ✓ `# Sparse storage: nearly all off-diagonal weights are zero.`

- ✗ `# Formerly a dense loop; switched to this for speed`
- ✓ usually *no comment* — the code stands on its own. Add one only if a future
  editor would otherwise reintroduce the slow form, and then state *why the fast
  form is required*, not what it replaced.

- ✗ `# as planned in the design doc, normalize before the fit`
- ✓ `# Normalize first: the solver assumes unit-scaled columns.`

Corollaries:

- **Never reference a planning artifact, chunk ID, session note, or "as
  planned".** A planning document is fair to cite only when it is a durable,
  committed file *in the repository itself*. A GitHub issue/PR number is fine as
  a terse pointer (`# see #123`) when the issue records context the code cannot
  — but it never substitutes for stating what the code does.
- **History lives in the commit log, not the source.** "Previously…",
  "Formerly…", "Regression:…", "this used to…" — drop them. The rare exception
  is when the history is the *only* thing that stops a future editor from
  re-making a mistake; then write the constraint as a present-tense fact ("must
  stay ≥ ε — zero triggers a singular solve"), not as a story about the past.
- **Intent and invariants, not motivation or biography.** No "motivating
  example", no roadmap, no "for now".
- **Match the surrounding code** in density, detail, and abstraction level.
  Sparse code gets sparse comments.
- **Write for a human**, not for an agent and not for yourself mid-task.

The same principle governs commit messages and docstrings: describe the change
and the invariant it establishes for a reader who has only the repository.

This applies with special force right after you have worked from a plan. The
plan, the session handoff, and the accumulated working-knowledge you just read
are scaffolding for *you*. The code, its comments, and the commit message must
read as if that scaffolding never existed.

## Julia generic indexing

Annotating an argument `AbstractArray` (or `AbstractVector`, etc.) is a
*promise* that the code works for any array — not just `Array`. That includes
arbitrary axes (`OffsetArray`), lazy wrappers (`view`, `PermutedDimsArray`,
`reshape`), GPU arrays, and arrays whose indices aren't `Int`. Either honor
that promise or annotate more narrowly. The cost of breaking it is silent wrong
answers for the very callers the broad annotation invited in.

The axiom that governs indexing is: `a = b[idxs]` implies `a[j] === b[idxs[j]]`.
So `a` inherits its *values* from `b` and its *axes* from `idxs` —
`axes(a) == axes(idxs)`, and `j` ranges over `eachindex(idxs)`. Most rules below
are corollaries: index with something whose axes you want the result to have.
The dual holds for assignment: `a[idxs] = b` sets `a[idxs[j]] = b[j]`.

Honoring it means writing against the data's *indices and axes*, never against
`1:length`:

- Iterate `eachindex(A)`, or `pairs(A)` when you need index-value pairs. Reserve
  `enumerate` for when you genuinely want a 1-based *counter* independent of A's
  keys — not as a stand-in for the index.
- To iterate several arrays together, `for i in eachindex(A, B)`: it both
  validates that they share indices and stays generic. One idiom replaces a
  manual length check plus `1:n`.
- Span one dimension with `axes(A, d)`, not `1:size(A, d)`. Use `firstindex` /
  `lastindex`, not `1` / `length`-as-`end`.
- Allocate index-matched results so axes and array type propagate:
  `similar(y, eltype(y), (axes(y, 1), Base.OneTo(k)))`, not
  `Matrix{T}(undef, length(y), k)`. Use `similar(y, T, ...)` to change eltype.
- Prefer `map` / broadcasting / comprehensions over index-matched inputs when
  you can — they already carry axes through correctly.
- Convert between linear and Cartesian indices with `LinearIndices(A)` /
  `CartesianIndices(A)`, not hand-rolled `(i-1)*n + j` arithmetic.

Check consistency at the top of the function and fail fast:

    axes(A) == axes(B) || throw(DimensionMismatch("A and B must match: $(axes(A)) vs $(axes(B))"))

If you deliberately write 1-based code — sometimes the honest choice — *declare*
it rather than assuming silently:

    Base.require_one_based_indexing(A, B)

An `OffsetArray` caller then gets a clear, immediate error instead of a wrong
result. Declaring the assumption is acceptable; leaving it implicit is not.

This is a correctness property, not a stylistic nicety: it should be enforced by
tests that run the package's entry points on `OffsetArray`- and `view`-wrapped
inputs and assert the results match (and that output axes track input axes).

## Julia @inbounds

The default is **no `@inbounds`**. Its downside is not a wrong-but-visible
answer — it is silent undefined behavior: an out-of-range access under
`@inbounds` reads or writes arbitrary memory instead of throwing a
`BoundsError`. That is the sharpest possible violation of fail-fast, so the
annotation has to earn its place rather than be added by reflex.

Most of the time you don't need it. Iterating `eachindex(A)` (see generic
indexing above) already lets the compiler prove the accesses are in-bounds and
elide the checks itself — you get the speed without the unsafety. The same
`eachindex`/`axes` habit that makes code generic also makes bounds-check
elision automatic. Reach for the annotation only when that hasn't happened.

`@inbounds` is *not* a "make it faster" button:

- It can make code **slower**. The bounds check it removes is often already
  free or already elided, and `@inbounds` can block LLVM transformations
  (vectorization among them) that would otherwise fire.
- It can make code **wrong** — silently, per the above.

So add it only when *all* of these hold, and prefer to leave it out when in
doubt:

- profiling shows the bounds check is a real bottleneck;
- the compiler genuinely cannot prove safety on its own (i.e. `eachindex`-style
  iteration did not already elide it);
- the in-bounds property is locally provable *and* covered by a test.

When it is warranted:

- wrap the **minimal** expression, never a whole function body;
- never apply it to an index derived from arithmetic or user input without a
  preceding guard;
- for a custom `getindex`, express the contract with `@boundscheck` and
  `Base.@propagate_inbounds` rather than scattering `@inbounds` at call sites.

A new `@inbounds` appearing in a diff is exactly the kind of unjustified
complexity a code review should flag and ask to see a benchmark for.

## TopOpt.jl-specific coding notes

- The package uses `PseudoDensities` (see `src/TopOpt.jl`) with type parameters
  tracking interpolation/penalization/filtering state. Preserve these type
  parameters when constructing or passing density arrays; do not silently
  strip them to plain `Vector`s.
- FEA assembly distinguishes penalized loads (`weights`/`fes`) from
  non-penalized loads (`dloads`/`fixedload`/`cload`). External loads are
  independent of material density — do not penalize them.
- Functions in `src/Functions/` extend `AbstractFunction` from Nonconvex.jl and
  must remain differentiable via Zygote. Avoid side effects or in-place
  mutation that breaks AD.
- Run JuliaFormatter before committing:
  `julia --project=. -e "using JuliaFormatter; format(\".\")"`.