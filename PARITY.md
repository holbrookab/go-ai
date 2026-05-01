# packages/ai Semantic Parity

This tracks semantic parity between upstream Vercel AI SDK `packages/ai/src` and this repo's Go package at `packages/ai`.

Inspected upstream: `/Users/dholbrook/src/ai/packages/ai/src`
Inspected Go package: `/Users/dholbrook/src/go-ai/packages/ai`

Status legend:

- `done`: Go package has a comparable public contract and behavior for the upstream feature area.
- `partial`: Go package has meaningful coverage, but important upstream semantics, options, events, or helpers are absent.
- `missing`: No comparable Go implementation yet.
- `n/a-go`: Intentional TypeScript/browser/runtime surface that does not map cleanly to a Go package.

## Current Go Surface

The Go package currently contains a compact language-model core:

- Prompt/message normalization for system/user/assistant/tool messages.
- Content parts for text, files, reasoning, reasoning files, tool calls, tool results, and sources.
- `GenerateText` with multi-step tool loops, stop conditions, tool choice filtering, active-tool filtering, provider-executed tool result tracking, input validation, tool-call repair, approval denial/user-approval outputs, retries, request timeouts, provider options, usage aggregation, callbacks, response metadata, custom download hooks, and text/json/object output strategies.
- `StreamText` with provider stream consumption, step accumulation, stream transforms, smooth streaming, streamed partial output parsing, streamed tool execution, streamed tool-call repair, and follow-up tool-loop steps.
- Lifecycle callbacks and telemetry hooks for text, stream text, object generation, and embeddings.
- `GenerateObject` / `StreamObject` with JSON response-format steering, object/array/enum/no-schema modes, upstream-style array output wrapping/unwrapping, JSON instruction helper, repair hook, best-effort schema validation, typed no-object errors, and repaired partial object streaming.
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
| `types` | partial | Language, embedding, image, speech, transcription, video, and reranking model/provider contracts; usage/warning/finish reason/metadata/response format/stream part/provider options, provider references, and file data contracts. | Missing request/response metadata variants, JSON value/schema typing, full warning taxonomy, and some provider-utils compatibility shims. |
| `prompt` | partial | Prompt standardization, message roles, content parts, tool-result ordering checks, call options, timeout config, typed role/conversion errors, file/reasoning-file validation, provider-reference file parts, base64/byte data-content normalization, supported-URL pass-through, unsupported-URL download fallback, custom download hook, and gateway error wrapping. | Missing full upstream model-message schema coverage, tool preparation edge semantics, and timeout helper functions with upstream naming/behavior. |
| `generate-text` | partial | `GenerateText`, `StreamText`, step results, tool calls/results, stop conditions, prepare-step hook, provider options, usage/warnings aggregation, streamed tool execution and follow-up steps; lifecycle callbacks/telemetry; active-tool filtering, stream transform hooks, `SmoothStream`, text/json/object output strategies, partial output on stream chunks, typed output decoding helper, text/reasoning extraction, message pruning, approval resolution helpers, and tool-call repair in generate/stream paths. | Missing full TS-style transforms over SDK-injected chunks, array/choice output strategies and element streams, separate partial-output stream API, complete abort/raw-chunk semantics, and exact promise/type-inference ergonomics that do not map directly to Go. |
| `error` | partial | `SDKError` plus typed named errors/guards for API calls, downloads, retries, invalid data content, invalid message role, message conversion, invalid stream part, invalid tool approval, approval tool-call-not-found, tool repair, unsupported model version, no object/image/speech/transcript/video/output, UI message stream, missing tool results, and no-such-tool. | Missing exact upstream provider/provider-utils message text and marker semantics for every edge. |
| `util` | partial | Partial JSON/fix JSON, cosine similarity, deep equality, string hash, public media/data URL normalization and SSRF-aware download, header/default merging, deep object merging, retry error/backoff with retry-header support, reader consumption, serial job executor, stitchable stream, and HTTP stream response helpers. | Need async iterable stream facade, merge callbacks/signals, stream simulation helpers, and runtime detection. |
| `logger` | partial | Warning logger interface, configurable logger, nil-safe text warning logger, first-warning info behavior, warning filters, compatibility warning helper, and warning logging from generation/embed/upload paths. | Need exact upstream warning-message text coverage for every provider-utils warning variant. |
| `registry` | done | `ProviderRegistry`, `CustomProvider`, language/embedding/image/video/speech/transcription/reranking model reference resolution, no-such-provider/no-such-model errors, custom separators, fallback provider behavior, files/skills API resolution. | n/a-go: TS compile-time experimental aliases and typed string-literal registry ergonomics do not map to Go. |
| `model` | done | Direct model-or-`provider:model` resolver helpers for language, embedding, image, video, speech, transcription, and reranking models; reusable mock model implementations for all current model families. | n/a-go: upstream `asLanguageModel`/`asEmbeddingModel`/`asImageModel`/`asSpeechModel`/`asTranscriptionModel`/`asVideoModel`/`asRerankingModel` adapters bridge TS provider specification versions; Go has one provider-facing interface per model family instead. |
| `middleware` | done | Language, embedding, image, video, speech, transcription, reranking, and provider wrappers; default language and embedding settings with header and deep provider-option merge; generate-side extract JSON/reasoning middleware; simulate streaming middleware with upstream start/metadata/start-delta-end/finish shape; add tool input examples middleware. | n/a-go: TS middleware `specificationVersion` markers and type-level transform overloads do not map to Go interfaces. |
| `telemetry` | partial | Generic event type, telemetry recorder interface, diagnostic-style event names/operation IDs, telemetry options for enable/input/output/function/filter controls, lifecycle callbacks/telemetry hooks for text, stream text, embeddings, object generation, media, uploads, and bounded language-model call start/end events for text generation/streaming. | Need full TS telemetry dispatcher/registry, diagnostic-channel publisher equivalents, inner embed/rerank model-call telemetry, and tool execution context telemetry wrapping. |
| `embed` | done | `Embed`, `EmbedMany`, embedding result/usage, max-values chunking, model serial/parallel capability, provider options, retries, warnings/metadata/responses, lifecycle callbacks/telemetry. | Provider-specific embedding implementations are outside `packages/ai` (n/a-go for core parity). |
| `rerank` | done | Reranking model contract and `Rerank` delegation API with original/reranked document views, ranking normalization, warnings/metadata/usage/response, lifecycle callbacks/telemetry, and invalid-index validation. | Provider implementations are outside `packages/ai` (n/a-go for core parity). |
| `generate-object` | partial | `GenerateObject`, `StreamObject`, JSON response format steering, object/array/enum/no-schema modes, upstream-style array output wrapping/unwrapping, `InjectJSONInstruction`, custom download hook, parsing, repair text, no-object typed errors, lifecycle callbacks/telemetry, best-effort JSON Schema validation, stricter mode validation, final stream validation errors, and repaired partial object stream events. | Missing full JSON Schema validation, complete output strategy abstraction/element streams, and upstream object promise/partial stream split semantics. |
| `generate-image` | done | Image model contract, generated file type, `GenerateImage`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-image typed error. | Provider implementations are outside `packages/ai`; deprecated TS aliases are n/a-go for core parity. |
| `generate-speech` | done | Speech model contract, generated audio file type, `GenerateSpeech`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-speech typed error. | Provider implementations are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `transcribe` | done | Transcription model contract, segment type, `Transcribe`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-transcript typed error. | Provider implementations are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `generate-video` | done | Video model contract, generated file type, `GenerateVideo`, warnings/metadata/response, lifecycle callbacks/telemetry, and no-video typed error. | Provider implementations and provider-specific polling are outside `packages/ai`; experimental TS aliases are n/a-go for core parity. |
| `upload-file` | done | Files API/provider contracts, upload data normalization for bytes/base64/text, media type detection, retries, warnings, request/response metadata, provider references, and lifecycle callbacks/telemetry. | Provider implementations and provider-specific file-part integration are outside `packages/ai` (n/a-go for core parity). |
| `upload-skill` | done | Skills API/provider contracts, multi-file skill upload delegation, retries, warnings, metadata, provider references, and lifecycle callbacks/telemetry. | Provider implementations and provider-specific manifest validation are outside `packages/ai` (n/a-go for core parity). |
| `text-stream` | partial | Go HTTP `ResponseWriter` helpers for text and data stream responses, text/completion pipe helpers, response constructors, status text, and default header preservation. | Need richer abort propagation for writer-based responses and exact Node/Web stream helper parity where it maps to Go. |
| `ui-message-stream` | partial | `UIMessageChunk`, chunk helpers, data/start/finish/error guards, chunk validation helper, response UI message ID selection, channel-based `CreateUIMessageStream` with write/merge, finish/step callbacks, abort tracking for finish callbacks, stream collector, JSON-to-SSE helpers, SSE/JSONL `http.ResponseWriter` piping, response constructors, default stream headers, and stream consumption hooks. | Need richer cancel/backpressure semantics, exact schema validation messages, and broader fixture coverage for message assembly. |
| `ui` | partial | `UIMessage` / `UIPart`, UI part helpers, `ValidateUIMessages`, `SafeValidateUIMessages`, schema-backed metadata/data validation hooks, static tool input/output validation, `ConvertToModelMessages`, pragmatic `AppendResponseMessages`, text stream processing, text-to-UI-message stream conversion, UI chunk application into assistant message state with stream-time metadata/data validation, transient/persistent data callbacks, server-side chat/completion request decoding and response streaming helpers, and approval-response completion reconciliation; supports text, file, reasoning, reasoning-file, provider-reference file parts, data-part conversion hooks, static/dynamic tools, tool results, step splitting, provider metadata, incomplete-tool-call filtering, and approval/output state updates. TS/browser-only hooks and chat transports (`useChat`, `useCompletion`, `DefaultChatTransport`, HTTP/text/direct browser transports) are n/a-go except for server/HTTP primitives. | Need exact upstream validation messages, dynamic tool schema validation, reconnect/resume stream server semantics, and broader fixture coverage. |
| `agent` | partial | `Agent` interface, `ToolLoopAgent` with settings, default `StepCount(20)`, prepare-call hook, active-tool filtering, merged callbacks including tool execution start/end, output/download forwarding, generate/stream methods, and `CreateAgentUIStream` bridge to UI chunks. | Need full call-options schema validation, runtime context/sensitive telemetry, direct stream-transform forwarding, agent UI stream response helpers, and inferred typing equivalents where meaningful in Go. |
| `test` | done | Reusable mock provider plus language/embedding/image/speech/transcription/video/reranking model helpers with call recording and default responses. | n/a-go: upstream mock server response helpers target Node/Web stream response objects; Go tests use `httptest.ResponseRecorder` and package HTTP helpers directly. |

## Cross-Cutting Parity Requirements

- Preserve upstream behavior, not only names. Each port should be tested against upstream semantics: prompt validation, tool-call ordering, stop conditions, usage aggregation, warning propagation, error shape, provider metadata, and streaming event order.
- Prefer Go-native public APIs where TypeScript-only ergonomics do not translate, but document intentional deviations in this file as they are made.
- Keep provider-facing interfaces stable before adding higher-level helpers. Most missing modules depend on complete model/type contracts.
- Streaming needs a shared Go abstraction before porting `streamText`, object streaming, UI streams, text streams, and agents. Avoid one-off stream shapes per feature.
- Error parity should be explicit. Add named error types/sentinels before features depend on them, so callers can inspect errors reliably.

## Recommended Port Order

1. **Core contracts and errors**
   - Expand `types`, `prompt`, and `error` first.
   - Complete request/response metadata variants and provider-reference contracts for all model families.
   - Add named error types/sentinels and prompt/data-content/message-conversion helpers.

2. **Utilities, retries, logging, and telemetry foundations**
   - Port public `util` pieces needed by generation APIs: retries/backoff, headers, stream consumption, partial JSON/fix JSON, download, cosine similarity, deep equality.
   - Add warning logging and telemetry registry/diagnostic hooks.

3. **Complete `generate-text` semantics**
   - Bring `StreamText` up to upstream behavior with chunk/event processing, transforms, callbacks, tool execution from streams, active tools, tool repair, output helpers, prune/smooth stream, and response-message conversion.
   - Fill remaining stop condition and prepare-step behavior gaps.

4. **Provider registry, model adapters, and middleware**
   - Port registry/custom provider behavior.
   - Add model version adapters/resolution.
   - Add wrappers/default-settings/extract/simulate middleware once interfaces are complete.

5. **Embeddings and reranking**
   - Harden `embed`, `embedMany`, and `rerank` against upstream tests and add provider implementations.

6. **Structured object generation**
   - Port remaining `generateObject` and `streamObject` strategy details, including full schema validation, element streams, partial streaming parity, repair text, and object-specific errors.

7. **Media and uploads**
   - Port image, speech, transcription, video, upload-file, and upload-skill APIs after the media model interfaces are stable.

8. **HTTP and UI stream helpers**
   - Port `text-stream` and `ui-message-stream` using Go HTTP/SSE primitives.
   - Then port UI message validation/conversion and chat transport/server-side equivalents.

9. **Agents**
   - Port agent/tool-loop behavior last, after text streaming, UI streams, tool approval/output handling, and model adapters are in place.

10. **Reusable test mocks**
    - Build reusable mocks alongside each model family, then consolidate once all provider/model interfaces settle.

## Immediate Next Slice

The next implementation slice should target the largest remaining user-visible `packages/ai` gaps:

- Add array/choice output strategies, object element streams, and separate partial-output stream APIs where they make sense in Go.
- Add reconnect/resume stream server semantics and broader upstream UI/chat fixtures.
- Tighten exact upstream validation/warning/error message parity where callers may assert message text.
- Decide whether agent call options should expose stream-transform knobs directly or keep them as lower-level text options.
