# go-ai

A Go port of the text core from Vercel's AI SDK, focused on a small v0 surface:

- `packages/ai`: text generation, streaming, prompt normalization, tool loops, retries, stop conditions, and shared model/provider contracts.
- `packages/bedrock`: Amazon Bedrock Converse text provider.
- `packages/vertex`: Google Vertex AI Gemini text provider.

Embeddings, images, video, files, UI streams, React/RSC helpers, agents, and Gateway are intentionally out of scope for this first slice.

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

