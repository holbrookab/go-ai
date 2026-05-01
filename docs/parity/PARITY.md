# Active Parity Backlog

This is the active work board for outstanding parity items. It intentionally excludes completed work and broad inventory. For the full snapshot, see `AUDIT.md`.

Baseline: `ai@7.0.0-beta.115`, upstream commit `9106864812053f39be867e0dff9af4bcb8cef2a6`.

## Queue

| Priority | Area | Label | Work item | Done when |
| --- | --- | --- | --- | --- |
| P1 | `telemetry` | go-native | Add a Go-native telemetry dispatcher/registry and tool execution context events. | Telemetry can be installed globally or per call, language/tool/model/tool-execution events carry filtered attributes, and TS diagnostic-channel publisher behavior is either mapped to Go or marked `n/a-go`. |
| P1 | `util` | go-native | Add callback/context composition and stream simulation helpers. | Go helpers cover callback merging, context cancellation composition, simulated streams, and current retry/stitchable helpers have upstream-style tests. |
| P2 | `error` | fixture-needed | Tighten exact error taxonomy and message text coverage. | Named error guards exist for provider/provider-utils errors and fixture tests assert message/cause/status fields for edge cases callers may branch on. |
| P2 | `logger` | fixture-needed | Cover all warning variants and formatting. | Warning fixtures cover unsupported, compatibility, deprecated, other, unknown warning payloads, filtering, and first-warning behavior. |
| P2 | `text-stream` | fixture-needed | Broaden HTTP response fixture coverage. | Tests cover headers, status text, flush behavior, error propagation, context cancellation, text stream, data stream, completion stream, and UI stream interop. |
| P2 | `ui` / chat server | fixture-needed | Port broader upstream chat/UI fixtures. | Fixture tests cover request decoding, message conversion, tool approval/output states, transient data, metadata/data schemas, and server response streams. |
| P2 | `generate-text` | go-native | Decide final Go shape for separate partial-output streams. | Either a separate channel/API is implemented, or `StreamPart.PartialOutput` and `element` stream parts are documented as the Go-native equivalent with tests. |
| P3 | `index.ts`, `global.ts` | n/a-go | Document TS module/global behavior as intentionally absent. | Browser/global setup and re-export ergonomics are documented as `n/a-go`; root README documents the Go import surface. |
| P3 | browser transports/hooks | n/a-go | Mark React/browser hooks and browser transports as `n/a-go`. | `useChat`, `useCompletion`, browser `DefaultChatTransport`, and direct browser fetch helpers are listed as `n/a-go` with server helper replacements named. |
| P3 | TS type inference | n/a-go | Document compile-time inference differences. | TS-only type assertions and type-level tests have Go equivalents or are marked `n/a-go`. |

## Current Blockers

None.

