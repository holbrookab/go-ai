# Changelog

## 0.2.2 - 2026-05-05

- Fixed Vertex message mapping so provider-facing messages use `Message.Text` as a fallback and skip empty content entries instead of sending empty `parts` arrays.

## 0.2.1 - 2026-05-05

- Aligned tool approval behavior with AI SDK semantics.

## 0.2.0 - 2026-05-01

- Expanded `packages/ai` toward Vercel AI SDK parity, including object generation, embeddings, agents, UI message streams, middleware, richer error types, media/upload helpers, and stricter schema validation.
- Added and broadened Bedrock and Vertex provider behavior, tests, and stream/raw chunk handling.
- Moved parity tracking into `docs/parity` with separate upstream baseline, active backlog, and audit snapshot docs.
- Added Apache-2.0 licensing with Vercel AI SDK attribution.

## 0.1.0

- Initial Go port of the core text generation and streaming surface.
