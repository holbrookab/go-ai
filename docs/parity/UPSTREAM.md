# Upstream Baseline

This repo currently tracks a local checkout of Vercel AI SDK.

| Field | Value |
| --- | --- |
| Upstream checkout | adjacent checkout at `../ai`; absolute local paths are intentionally not tracked |
| Upstream package version | `ai@7.0.0-beta.115` |
| Upstream commit | `9106864812053f39be867e0dff9af4bcb8cef2a6` |
| Upstream working tree | clean when checked |
| Checked on | 2026-05-01 |
| Go module | `github.com/holbrookab/go-ai` |

## Compared Scope

Primary scope:

- `../ai/packages/ai/src`
- `../ai/packages/amazon-bedrock`
- `../ai/packages/google-vertex`
- shared provider/provider-utils behavior as needed by the Go public surface

Current Go implementation scope:

- `packages/ai`
- `packages/bedrock`
- `packages/vertex`
- `internal/httputil`
- `internal/retry`
- `packages/bedrock/internal/sigv4`

## Tracking Rules

- `UPSTREAM.md` records the comparison baseline, not outstanding work.
- `PARITY.md` records outstanding work only.
- `AUDIT.md` records the broader state of the port, including completed and Go-native areas.
- If upstream is bumped, update this file first, then refresh `AUDIT.md`, then create or adjust active backlog rows in `PARITY.md`.

## Known Go-Native Differences

- TypeScript compile-time inference tests are replaced by Go compile/runtime tests and examples.
- React, RSC, browser transports, and hooks are not runtime Go packages. Server-side HTTP helpers and UI stream primitives are the Go replacement surface.
- Node/Web stream helpers map to Go channels, `io.Reader`, `http.Response`, and `http.ResponseWriter` helpers.
