# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

- **Agent meta instructions**: [`AGENTS.md`](AGENTS.md) is the source of truth
  for stance, Memory Bank protocol, Julia development and style guide, and
  devops rules. It also links the mode-specific rules in `.roo/rules-<mode>/`:
  - [`.roo/rules-code/AGENTS.md`](.roo/rules-code/AGENTS.md) — Code mode
  - [`.roo/rules-debug/AGENTS.md`](.roo/rules-debug/AGENTS.md) — Debug mode
  - [`.roo/rules-architect/AGENTS.md`](.roo/rules-architect/AGENTS.md) —
    Architect mode
  - [`.roo/rules-ask/AGENTS.md`](.roo/rules-ask/AGENTS.md) — Ask mode
- **Project description**: the detailed project specifics (architecture, module
  structure, build/test commands, dependencies, usage examples) live in the
  **Memory Bank** at [`memory-bank/`](memory-bank/) (git-ignored, kept up to
  date). Read those files for project context.