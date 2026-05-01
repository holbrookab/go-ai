# go-ai

A Go port of Vercel's AI SDK, with the public surface organized under `packages/*` to match the upstream TypeScript repo where that shape makes sense in Go.

- `packages/ai`: generation, streaming, object generation, embeddings, tool loops, agents, UI message streams, middleware, prompt normalization, retries, stop conditions, and shared model/provider contracts.
- `packages/bedrock`: Amazon Bedrock Converse provider.
- `packages/vertex`: Google Vertex AI Gemini provider.

## Parity Tracking

Parity work lives in [`docs/parity`](docs/parity/README.md). That directory is the source of truth for what this port has been compared against and what remains:

- [`docs/parity/UPSTREAM.md`](docs/parity/UPSTREAM.md): upstream AI SDK version, commit, local path, and comparison scope.
- [`docs/parity/PARITY.md`](docs/parity/PARITY.md): active backlog only. Treat it like the JIRA board for unfinished parity work.
- [`docs/parity/AUDIT.md`](docs/parity/AUDIT.md): broader snapshot of implemented, Go-native, fixture-needed, and intentionally non-Go surfaces.

When updating parity, refresh `UPSTREAM.md` first, update `AUDIT.md` as the broad reference, and keep `PARITY.md` limited to actionable outstanding work. Completed rows should come out of `PARITY.md` instead of accumulating as history.

## Generate Text

```go
package main

import (
	"context"
	"fmt"

	"github.com/holbrookab/go-ai/packages/ai"
	"github.com/holbrookab/go-ai/packages/vertex"
)

func main() {
	provider := vertex.New(vertex.Settings{
		Project:  "my-project",
		Location: "global",
	})

	result, err := ai.GenerateText(context.Background(), ai.GenerateTextOptions{
		Model:  provider.LanguageModel("gemini-2.5-flash"),
		System: "You are concise.",
		Prompt: "Say hello from Go.",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Text)
}
```

## Tool Loop

Use `StopWhen: []ai.StopCondition{ai.LoopFinished()}` to allow the model to call client-side tools until the model stops asking for tools.

```go
result, err := ai.GenerateText(ctx, ai.GenerateTextOptions{
	Model:    model,
	Prompt:   "What is the weather in NYC?",
	StopWhen: []ai.StopCondition{ai.LoopFinished()},
	Tools: map[string]ai.Tool{
		"weather": {
			Description: "Get the weather for a city.",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []string{"city"},
			},
			Execute: func(ctx context.Context, call ai.ToolCall, opts ai.ToolExecutionOptions) (any, error) {
				return "sunny", nil
			},
		},
	},
})
```

## Provider Auth

Bedrock auth follows the upstream SDK precedence: explicit API key, `AWS_BEARER_TOKEN_BEDROCK`, then SigV4 via credential provider, explicit keys, or AWS environment variables.

Vertex auth uses `GOOGLE_VERTEX_API_KEY` or explicit API key for Express Mode. Otherwise it uses an injected token source, `GOOGLE_VERTEX_ACCESS_TOKEN`, service account JSON from `GOOGLE_APPLICATION_CREDENTIALS`, or the metadata server.

## License

This project is licensed under Apache-2.0. Portions are derived from Vercel AI SDK; see [LICENSE](LICENSE) for attribution.
