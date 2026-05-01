# Parity Audit Snapshot

This is the broad audit snapshot for the Go port. It is reference material, not the active backlog. Active work belongs in `PARITY.md`.

Baseline: `ai@7.0.0-beta.115`, upstream commit `9106864812053f39be867e0dff9af4bcb8cef2a6`.

## Current Go Surface

The Go package currently contains:

- Prompt/message normalization for system/user/assistant/tool messages.
- Content parts for text, files, reasoning, reasoning files, tool calls, tool results, and sources.
- `GenerateText` with multi-step tool loops, stop conditions, tool choice filtering, active-tool filtering, provider-executed tool result tracking, input validation, tool-call repair, approval denial/user-approval outputs, retries, request timeouts, provider options, usage aggregation, callbacks, response metadata, custom download hooks, and text/json/object/array/choice output strategies.
- `StreamText` with provider stream consumption, step accumulation, stream transforms, smooth streaming, streamed partial output parsing, array element events, abort/raw-chunk events, streamed tool execution, streamed tool-call repair, and follow-up tool-loop steps.
- Lifecycle callbacks and telemetry hooks for text, stream text, object generation, embeddings, media, uploads, and model calls.
- `GenerateObject` / `StreamObject` with JSON response-format steering, object/array/enum/no-schema modes, array output wrapping/unwrapping, JSON instruction helper, repair hook, best-effort schema validation, typed no-object errors, repaired partial object streaming, and Go-native array element streaming.
- `Embed` / `EmbedMany`, `Rerank`, `GenerateImage`, `GenerateVideo`, `GenerateSpeech`, `Transcribe`, `UploadFile`, and `UploadSkill` core delegation APIs.
- Provider/model interfaces for language, embedding, image, video, speech, transcription, reranking, file uploads, and skill uploads.
- Bedrock and Vertex text providers.
- Tool definitions with input/output schemas, input examples, provider tools, execution, validation, custom model output, approval hooks, and tool execution context.
- UI message conversion/validation, UI stream processing/responses, chat/completion server helpers, text stream helpers, reusable mock models, and `ToolLoopAgent`.

## Upstream Folder Snapshot

| Upstream area | Status | Notes |
| --- | --- | --- |
| `index.ts`, `global.ts`, top-level re-exports | partial | Single Go package exports core types/functions directly. Gateway/global/browser setup remains a Go-native documentation item. |
| `types` | partial | Model/provider contracts, usage, warnings, metadata, response formats, request/response metadata, JSON value/schema aliases, and file data contracts exist. Warning taxonomy and provider-utils shims need fixture hardening. |
| `prompt` | partial | Prompt standardization, message roles, stricter tool-call/tool-result validation, tool-result ordering, file normalization/download, active/named tool validation, and tool context propagation exist. Remaining work is mostly exact helper naming/fixtures. |
| `generate-text` | partial | Text generation/streaming, tool loops, transforms, smooth streaming, output strategies, partial output, element events, abort/raw chunks, callbacks, telemetry hooks, and repair paths exist. Separate TS-style partial-output streams are a Go-native decision item. |
| `generate-object` | partial | Object/array/enum/no-schema modes, stream object, partial object, element streams, repair, and best-effort schema validation exist. Full JSON Schema parity remains intentionally incremental. |
| `error` | partial | Named SDK errors and guards exist. Exact upstream message/marker fixture coverage remains outstanding. |
| `util` | partial | Partial JSON, cosine similarity, deep equality, hashing, media/data URL normalization, SSRF-aware download, header/default merging, deep merge, retry/backoff, reader consumption, serial executor, stitchable streams, and HTTP helpers exist. Callback/signal composition and stream simulation helpers remain. |
| `logger` | partial | Warning logger surface exists. Broader warning fixture coverage remains. |
| `telemetry` | partial | Generic event recorder and lifecycle/model-call hooks exist. Dispatcher/registry and tool execution context telemetry remain. |
| `registry` | done | Provider registry, custom provider, model references, fallback behavior, files/skills APIs, and no-such errors exist. |
| `model` | done | Direct model or `provider:model` resolvers and mock models exist for current model families. |
| `middleware` | done | Language/embedding/image/video/speech/transcription/reranking/provider wrappers and core middleware helpers exist. |
| `embed` | done | Core embedding APIs exist; provider-specific embeddings are outside `packages/ai`. |
| `rerank` | done | Core rerank API exists; provider implementations are outside `packages/ai`. |
| `generate-image` | done | Core image delegation API exists; provider implementations are outside `packages/ai`. |
| `generate-speech` | done | Core speech delegation API exists; provider implementations are outside `packages/ai`. |
| `transcribe` | done | Core transcription delegation API exists; provider implementations are outside `packages/ai`. |
| `generate-video` | done | Core video delegation API exists; provider polling implementations are outside `packages/ai`. |
| `upload-file` | done | Files API/provider contract and upload delegation exist. |
| `upload-skill` | done | Skills API/provider contract and upload delegation exist. |
| `text-stream` | partial | Go HTTP response helpers exist. Broader fixture coverage remains. |
| `ui-message-stream` | partial | UI chunks, validation, response IDs, context-aware write/merge, finish callbacks, SSE/JSONL responses, terminal error chunks, and stream consumption hooks exist. Exact validation text and fixtures remain. |
| `ui` | partial | UI message validation/conversion/processing, static/dynamic tool schemas, data/metadata schemas, server chat/completion helpers, approval reconciliation, and resume callback handling exist. Broader fixtures remain. |
| `agent` | partial | `ToolLoopAgent`, default step limit, prepare-call hook, active-tool filtering, callbacks, output/download/transform forwarding, and UI bridge exist. Runtime context/sensitive telemetry and response helpers remain. |
| `test` | done | Mock provider and model helpers exist; Go tests use `httptest` and package HTTP helpers. |

## Cross-Cutting Rules

- Preserve behavior, not only names.
- Prefer Go-native APIs where TypeScript, browser, or React ergonomics do not translate.
- Keep provider-facing interfaces stable before adding higher-level helpers.
- Add named error types before features depend on them.
- Keep the active backlog in `PARITY.md` focused on unfinished work only.

