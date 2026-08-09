# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

- **Agent meta instructions**: [`AGENTS.md`](AGENTS.md) is the source of truth
  for stance, Memory Bank protocol, Julia development and style guide, and
  devops rules. It also links the topic-specific guidance under
  `.opencode/instructions/`:
  - [`.opencode/instructions/code.md`](.opencode/instructions/code.md) — code
    guidance
  - [`.opencode/instructions/debug.md`](.opencode/instructions/debug.md) —
    debugging guidance
  - [`.opencode/instructions/architect.md`](.opencode/instructions/architect.md)
    — architecture guidance
  - [`.opencode/instructions/ask.md`](.opencode/instructions/ask.md) —
    explanation guidance
- **Project description**: the detailed project specifics (architecture, module
  structure, build/test commands, dependencies, usage examples) live in the
  **Memory Bank** at [`memory-bank/`](memory-bank/) (git-ignored, kept up to
  date). Read those files for project context.