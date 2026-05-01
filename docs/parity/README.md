# Parity Tracking

This directory is the working parity system for the Go port of Vercel's AI SDK.

Use these files as separate sources of truth:

- `UPSTREAM.md`: the upstream baseline. It records exactly what SDK snapshot and local paths this repo was compared against.
- `PARITY.md`: the active backlog. Treat this like the JIRA board for outstanding parity work.
- `AUDIT.md`: the reference audit. It records the current broad package-by-package state so future agents can orient quickly without polluting the backlog.

## Workflow

When comparing against upstream:

1. Update `UPSTREAM.md` only when the upstream version, commit, source path, package scope, or comparison date changes.
2. Put only unfinished parity work in `PARITY.md`.
3. Keep completed or intentionally Go-native behavior in `AUDIT.md`, not the active board.
4. When a backlog row is finished, remove it from `PARITY.md` and update the relevant summary in `AUDIT.md`.
5. Every parity item should name acceptance criteria and the test evidence needed to close it.

## Status Labels

- `go-native`: needs an idiomatic Go equivalent rather than a literal TypeScript/browser port.
- `fixture-needed`: implementation exists, but upstream-style fixture coverage is not broad enough.
- `n/a-go`: browser, React, TypeScript type-system, or Node-specific behavior that should be documented as intentionally out of scope for Go.

`blocker` should appear only when the repo cannot reasonably claim semantic parity for the tracked scope until that item is solved. There are currently no active blocker rows.

