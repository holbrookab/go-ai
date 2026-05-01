# packages/ai Semantic Parity

This tracks semantic parity between upstream Vercel AI SDK `packages/ai/src` and this repo's Go package at `packages/ai`.

Inspected upstream: `/Users/dholbrook/src/ai/packages/ai/src`
Inspected Go package: `/Users/dholbrook/src/go-ai/packages/ai`

Status legend:

- `done`: Go package has a comparable public contract and behavior for the upstream feature area.
- `partial`: Go package has meaningful coverage, but important upstream semantics, options, events, or helpers are absent.
- `missing`: No comparable Go implementation yet.
- `n/a-go`: Intentional TypeScript/browser/runtime surface that does not map cleanly to a Go package.

Checklist labels:

- `blocker`: Must be implemented or consciously redesigned before claiming semantic parity.
- `go-native`: Needs a Go equivalent rather than a literal TypeScript/browser port.
- `fixture-needed`: Implementation exists, but upstream-style fixtures/tests are not broad enough.
- `n/a-go`: Browser, React, TypeScript type-system, or Node-specific behavior that should be documented as intentionally out of scope for Go.
- `done`: Implemented and tested enough for the current Go surface.

## Current Go Surface

The Go package currently contains a compact language-model core:

- Prompt/message normalization for system/user/assistant/tool messages.
- Content parts for text, files, reasoning, reasoning files, tool calls, tool results, and sources.
- `GenerateText` with multi-step tool loops, stop conditions, tool choice filtering, active-tool filtering, provider-executed tool result tracking, input validation, tool-call repair, approval denial/user-approval outputs, retries, request timeouts, provider options, usage aggregation, callbacks, response metadata, custom download hooks, and text/json/object/array/choice output strategies.
- `StreamText` with provider stream consumption, step accumulation, stream transforms, smooth streaming, streamed partial output parsing, array element events, abort/raw-chunk events, streamed tool execution, streamed tool-call repair, and follow-up tool-loop steps.
- Lifecycle callbacks and telemetry hooks for text, stream text, object generation, and embeddings.
- `GenerateObject` / `StreamObject` with JSON response-format steering, object/array/enum/no-schema modes, upstream-style array output wrapping/unwrapping, JSON instruction helper, repair hook, best-effort schema validation, typed no-object errors, repaired partial object streaming, and Go-native array element streaming.
- `Embed` / `EmbedMany` with embedding model contract, chunking, bounded parallelism, retries, ordered results, usage/warning/metadata aggregation.
- Provider/model interfaces for language, embedding, image, video, speech, transcription, reranking, file uploads, and skill uploads; a provider registry; language/embedding/image/provider middleware wrappers; model reference helpers; call options, warnings, usage, finish reasons, provider references, request/response metadata, and typed SDK errors.
- First-pass core functions for image, video, speech, transcription, and reranking that delegate to model contracts; provider implementations for those families are not present yet.
- Tool definitions with input/output schemas, input examples, provider tools, execution, validation, custom model output, and approval hooks.
- Public helper slices for warning logging, partial JSON, cosine similarity/deep equality/hash utilities, header/default merging, deep object merging, retry errors/backoff, reader consumption, serial job execution, stitchable streams, text/reasoning extraction, active-tool filtering, message pruning, media/data URL normalization/download, HTTP stream responses, chat/completion server helpers, UI message conversion/validation, UI message stream processing/responses, reusable mock models, and upload-file/upload-skill delegation.
- `ToolLoopAgent` and agent-to-UI-stream helpers built on top of `GenerateText` / `StreamText`.

There are no Go folders mirroring upstream modules yet; all current code is in `packages/ai/*.go`.

## Upstream Folder Parity

| Upstream folder/file | Status | Current Go coverage | Remaining semantic gaps |
| --- | --- | --- | --- |
| `index.ts`, `global.ts`, top-level re-exports | partial | Single Go package exports core types/functions directly. `Version` exists. | No gateway/provider-utils equivalents, no explicit module-level export audit, no global setup equivalent. |
| `types` | partial | Language, embedding, image, speech, transcription, video, and reranking model/provider contracts; usage/warning/finish reason/metadata/response format/stream part/provider options, provider references, file data contracts, explicit JSON value/schema aliases, and expanded request/response metadata fields. | Missing full warning taxonomy and some provider-utils compatibility shims. |
| `prompt` | partial | Prompt standardization, message roles, content parts, stricter tool-call/tool-result validation, orphan/missing tool-result ordering checks, call options, timeout config, typed role/conversion errors, file/reasoning-file validation, provider-reference file parts, base64/byte data-content normalization, supported-URL pass-through, unsupported-URL download fallback, custom download hook, schema-backed tool input validation, active/named tool validation, tools context propagation, and gateway error wrapping. | Missing timeout helper functions with exact upstream naming/behavior and broader fixture coverage for every upstream edge case. |
| `generate-text` | partial | `GenerateText`, `StreamText`, step results, tool calls/results, stop conditions, prepare-step hook, provider options, usage/warnings aggregation, streamed tool execution and follow-up steps; lifecycle callbacks/telemetry; active-tool filtering, stream transform hooks, `SmoothStream`, text/json/object/array/choice output strategies, partial output on stream chunks, Go-native array element stream parts, abort/raw provider chunk events, typed output decoding helper, text/reasoning extraction, message pruning, approval resolution helpers, and tool-call repair in generate/stream paths. | Missing full TS-style transforms over SDK-injected chunks, separate partial-output stream API, and exact promise/type-inference ergonomics that do not map directly to Go. |
| `error` | partial | `SDKError` plus typed named errors/guards for API calls, downloads, retries, invalid data content, invalid message role, message conversion, invalid stream part, invalid tool approval, approval tool-call-not-found, tool repair, unsupported model version, no object/image/speech/transcript/video/output, UI message stream, missing tool results, and no-such-tool. | Missing exact upstream provider/provider-utils message text and marker semantics for every edge. |
| `util` | partial | Partial JSON/fix JSON, cosine similarity, deep equality, string hash, public media/data URL normalization and SSRF-aware download, header/default merging, deep object merging, retry error/backoff with retry-header support, reader consumption, serial job executor, stitchable stream, and HTTP stream response helpers. | Need async iterable stream facade, merge callbacks/signals, stream simulation helpers, and runtime detection. |
| `logger` | partial | Warning logger interface, configurable logger, nil-safe text warning logger, first-warning info behavior, warning filters, compatibility warning helper, and warning logging from generation/embed/upload paths. | Need exact upstream warning-message text coverage for every provider-utils warning variant. |
| `registry` | done | `ProviderRegistry`, `CustomProvider`, language/embedding/image/video/speech/transcription/reranking model reference resolution, no-such-provider/no-such-model errors, custom separators, fallback provider behavior, files/skills API resolution. | n/a-go: TS compile-time experimental aliases and typed string-literal registry ergonomics do not map to Go. |
| `model` | done | Direct model-or-`provider:model` resolver helpers for language, embedding, image, video, speech, transcription, and reranking models; reusable mock model implementations for all current model families. | n/a-go: upstream `asLanguageModel`/`asEmbeddingModel`/`asImageModel`/`asSpeechModel`/`asTranscriptionModel`/`asVideoModel`/`asRerankingModel` adapters bridge TS provider specification versions; Go has one provider-facing interface per model family instead. |
| `middleware` | done | Language, embedding, image, video, speech, transcription, reranking, and provider wrappers; default language and embedding settings with header and deep provider-option merge; generate-side extract JSON/reasoning middleware; simulate streaming middleware with upstream start/metadata/start-delta-end/finish shape; add tool input examples middleware. | n/a-go: TS middleware `specificationVersion` markers and type-level transform overloads do not map to Go interfaces. |
| `telemetry` | partial | Generic event type, telemetry recorder interface, diagnostic-style event names/operation IDs, telemetry options for enable/input/output/function/filter controls, lifecycle callbacks/telemetry hooks for text, stream text, embeddings, object generation, media, uploads, and bounded language-model call start/end events for text generation/streaming. | Need full TS telemetry dispatcher/registry, diagnostic-channel publisher equivalents, inner embed/rerank model-call telemetry, and tool execution context telemetry wrapping. |
| `embed` | done | `Embed`, `EmbedMany`, embedding result/usage, max-values chunking, model serial/parallel capability, provider options, retries, warnings/metadata/responses, lifecycle callbacks/telemetry. | Provider-specific embedding implementations are outside `packages/ai` (n/a-go for core parity). |
| `rerank` | done | Reranking model contract and `Rerank` delegation API with original/reranked document views, ranking normalization, warnings/metadata/usage/response, lifecycle callbacks/telemetry, and invalid-index validation. | Provider implementations are outside `packages/ai` (n/a-go for core parity). |
| `generate-object` | partial | `GenerateObject`, `StreamObject`, JSON response format steering, object/array/enum/no-schema modes, upstream-style array output wrapping/unwrapping, `InjectJSONInstruction`, custom download hook, parsing, repair text, no-object typed errors, lifecycle callbacks/telemetry, best-effort JSON Schema validation, stricter mode validation, final stream validation errors, repaired partial object stream events, and array element stream events/channel. | Missing full JSON Schema validation and upstream object promise/partial stream split semantics. |
| `generate-image` | done | Image model contract, generated file type, `GenerateImage`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-image typed error. | Provider implementations are outside `packages/ai`; deprecated TS aliases are n/a-go for core parity. |
| `generate-speech` | done | Speech model contract, generated audio file type, `GenerateSpeech`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-speech typed error. | Provider implementations are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `transcribe` | done | Transcription model contract, segment type, `Transcribe`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-transcript typed error. | Provider implementations are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `generate-video` | done | Video model contract, generated file type, `GenerateVideo`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-video typed error. | Provider implementations and provider-specific polling are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `upload-file` | done | Files API/provider contracts, upload data normalization for bytes/base64/text, media type detection, retries, warnings, request/response metadata, provider references, and lifecycle callbacks/telemetry. | Provider implementations and provider-specific file-part integration are outside `packages/ai` (n/a-go for core parity). |
| `upload-skill` | done | Skills API/provider contracts, multi-file skill upload delegation, retries, warnings, metadata, provider references, and lifecycle callbacks/telemetry. | Provider implementations and provider-specific manifest validation are outside `packages/ai` (n/a-go for core parity). |
| `text-stream` | partial | Go HTTP `ResponseWriter` helpers for text and data stream responses, text/completion pipe helpers, response constructors, status text, and default header preservation. | Need richer abort propagation for writer-based responses and exact Node/Web stream helper parity where it maps to Go. |
| `ui-message-stream` | partial | `UIMessageChunk`, chunk helpers, data/start/finish/error guards, chunk validation helper, response UI message ID selection, channel-based `CreateUIMessageStream` with context-aware write/merge, finish/step callbacks, abort tracking for finish callbacks, stream collector, JSON-to-SSE helpers, SSE/JSONL `http.ResponseWriter` piping, terminal error chunk handling, response constructors, default stream headers, and stream consumption hooks. | Need exact upstream validation messages and broader fixture coverage for message assembly. |
| `ui` | partial | `UIMessage` / `UIPart`, UI part helpers, `ValidateUIMessages`, `SafeValidateUIMessages`, schema-backed metadata/data validation hooks, static and dynamic tool input/output validation, `ConvertToModelMessages`, pragmatic `AppendResponseMessages`, text stream processing, text-to-UI-message stream conversion, UI chunk application into assistant message state with stream-time metadata/data/tool validation, transient/persistent data callbacks, server-side chat/completion request decoding and response streaming helpers, Go-native resume-stream callback/204 handling, and approval-response completion reconciliation; supports text, file, reasoning, reasoning-file, provider-reference file parts, data-part conversion hooks, static/dynamic tools, tool results, step splitting, provider metadata, incomplete-tool-call filtering, and approval/output state updates. TS/browser-only hooks and chat transports (`useChat`, `useCompletion`, `DefaultChatTransport`, HTTP/text/direct browser transports) are n/a-go except for server/HTTP primitives. | Need exact upstream validation messages and broader fixture coverage. |
| `agent` | partial | `Agent` interface, `ToolLoopAgent` with settings, default `StepCount(20)`, prepare-call hook, active-tool filtering, merged callbacks including tool execution start/end, output/download/stream-transform forwarding, generate/stream methods, and `CreateAgentUIStream` bridge to UI chunks. | Need full call-options schema validation, runtime context/sensitive telemetry, agent UI stream response helpers, and inferred typing equivalents where meaningful in Go. |
| `test` | done | Reusable mock provider plus language/embedding/image/speech/transcription/video/reranking model helpers with call recording and default responses. | n/a-go: upstream mock server response helpers target Node/Web stream response objects; Go tests use `httptest.ResponseRecorder` and package HTTP helpers directly. |

## Cross-Cutting Parity Requirements

- Preserve upstream behavior, not only names. Each port should be tested against upstream semantics: prompt validation, tool-call ordering, stop conditions, usage aggregation, warning propagation, error shape, provider metadata, and streaming event order.
- Prefer Go-native public APIs where TypeScript-only ergonomics do not translate, but document intentional deviations in this file as they are made.
- Keep provider-facing interfaces stable before adding higher-level helpers. Most missing modules depend on complete model/type contracts.
- Streaming needs a shared Go abstraction before porting `streamText`, object streaming, UI streams, text streams, and agents. Avoid one-off stream shapes per feature.
- Error parity should be explicit. Add named error types/sentinels before features depend on them, so callers can inspect errors reliably.

## Parity Work Queue

This is the current hard queue for moving remaining `partial` rows to `done`, `go-native`, or `n/a-go`.

| Priority | Area | Label | Work item | Done when |
| --- | --- | --- | --- | --- |
| P0 | `generate-text` / `generate-object` | done | Add array and choice output strategies, including stream-time partial output behavior where useful. | `GenerateText` and `StreamText` support text/json/object/array/choice output shapes with focused tests; unsupported TS-only inference is documented as n/a-go. |
| P0 | `generate-object` | done | Add object element streams for array output or a Go-native channel equivalent. | Array object streaming can emit validated/new elements separately from whole-object partials, with tests based on upstream output-strategy behavior. |
| P0 | `streamText` | done | Model complete abort/raw-chunk semantics. | Aborted streams and raw provider chunks have predictable public events/results; tests cover mid-stream abort and raw chunk pass-through. |
| P0 | `agent` | done | Forward stream transforms and output settings through `ToolLoopAgent` stream/generate options consistently. | Agent options/settings/prepared-call structs can pass `Output`, `Download`, and stream transforms; tests prove forwarding through `Generate` and `Stream`. |
| P0 | `ui` / chat server | done | Add reconnect/resume stream server semantics. | Go chat helpers can resume from a known message/stream state with a resume callback and return `204` when no active stream exists; tests cover resume request parsing and response shape. |
| P1 | `prompt` | done | Complete remaining model-message schema validation and tool-preparation edge semantics. | Prompt conversion rejects/normalizes the upstream edge cases for system/user/assistant/tool roles, tool-result ordering, provider tools, dynamic tools, and file/reasoning-file parts. |
| P1 | `types` | done | Fill request/response metadata variants and JSON value/schema typing gaps. | Public types cover upstream metadata fields that providers need; JSON schema/value APIs are explicit and tests assert provider-facing call options. |
| P1 | `telemetry` | go-native | Add a Go-native telemetry dispatcher/registry and tool execution context events. | Telemetry can be installed globally or per call, language/tool/model call events carry filtered attributes, and TS diagnostic-channel publisher is marked n/a-go unless a Go equivalent is added. |
| P1 | `util` | go-native | Add merge callback/signal equivalents and stream simulation helpers. | Go helpers cover callback merging, context cancellation composition, simulated streams, and current retry/stitchable helpers have upstream-style tests. |
| P1 | `ui-message-stream` | done | Strengthen cancel/backpressure semantics. | Stream writers, merge, response helpers, and finish callbacks behave deterministically under cancellation, slow consumers, and error chunks. |
| P1 | `ui` | done | Add dynamic tool validation and tool output-schema parity. | Static and dynamic tool parts validate inputs/outputs where schemas exist, while upstream TODOs are documented explicitly. |
| P2 | `error` | fixture-needed | Tighten exact error taxonomy and message text. | Named error guards exist for provider/provider-utils errors and fixture tests assert message/cause/status fields for edge cases callers may branch on. |
| P2 | `logger` | fixture-needed | Cover all warning variants and formatting. | Warning fixtures cover unsupported, compatibility, deprecated, other, unknown warning payloads, filtering, and first-warning behavior. |
| P2 | `text-stream` | fixture-needed | Broaden HTTP response fixture coverage. | Tests cover headers, status text, flush behavior, error propagation, context cancellation, text stream, data stream, completion stream, and UI stream interop. |
| P2 | `ui` / chat server | fixture-needed | Port broader upstream chat/UI fixtures. | Fixture tests cover request decoding, message conversion, tool approval/output states, transient data, metadata/data schemas, and server response streams. |
| P2 | `generate-text` | go-native | Decide final Go shape for separate partial-output streams. | Either a separate channel/API is implemented, or `StreamPart.PartialOutput` is documented as the Go-native equivalent with tests. |
| P3 | `index.ts`, `global.ts` | n/a-go | Document TS module/global behavior as intentionally absent. | `PARITY.md` marks browser/global setup and re-export ergonomics as n/a-go; README documents Go package import surface. |
| P3 | browser transports/hooks | n/a-go | Mark React/browser hooks and browser transports as n/a-go. | `useChat`, `useCompletion`, browser `DefaultChatTransport`, and direct browser fetch helpers are listed as n/a-go with server helper replacements named. |
| P3 | TS type inference | n/a-go | Document compile-time inference differences. | TS-only type assertions and type-level tests have Go equivalents or are marked n/a-go. |

## Next Slices

Best next implementation slices:

1. `P1 telemetry`: dispatcher/registry plus tool execution context telemetry.
2. `P1 util streams`: callback/context composition and stream simulation helpers.
3. `P2 fixture hardening`: exact error/warning/text-stream/chat fixture coverage.
4. `P2 text-stream/chat`: broader HTTP response and UI/chat fixtures.
5. `P3 n/a-go docs`: module/global/browser/type-inference documentation.
