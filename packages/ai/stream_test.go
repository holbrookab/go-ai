package ai

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestStreamTextRunsToolLoop(t *testing.T) {
	calls := 0
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		calls++
		ch := make(chan StreamPart, 8)
		if calls == 1 {
			ch <- StreamPart{Type: "tool-call", ToolCallID: "call-1", ToolName: "weather", ToolInput: `{"city":"NYC"}`}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishToolCalls}}
		} else {
			if len(opts.Prompt) == 0 || opts.Prompt[len(opts.Prompt)-1].Role != RoleTool {
				t.Fatalf("second stream call should include tool result, got %#v", opts.Prompt)
			}
			ch <- StreamPart{Type: "text-delta", TextDelta: "sunny"}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}
	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:    model,
			Prompt:   "weather?",
			StopWhen: []StopCondition{LoopFinished()},
			Tools: map[string]Tool{
				"weather": {
					InputSchema: map[string]any{"type": "object"},
					Execute: func(ctx context.Context, call ToolCall, opts ToolExecutionOptions) (any, error) {
						return "sunny", nil
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	var text string
	var toolResultSeen bool
	for part := range result.Stream {
		if part.Type == "text-delta" {
			text += part.TextDelta
		}
		if part.Type == "tool-result" {
			toolResultSeen = true
		}
	}
	if text != "sunny" {
		t.Fatalf("expected sunny, got %q", text)
	}
	if !toolResultSeen {
		t.Fatalf("expected streamed tool result")
	}
	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}
}

func TestStreamTextRepairsUnavailableToolCall(t *testing.T) {
	calls := 0
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		calls++
		ch := make(chan StreamPart, 8)
		if calls == 1 {
			ch <- StreamPart{Type: "tool-call", ToolCallID: "call-1", ToolName: "whether", ToolInput: `{"city":"NYC"}`}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishToolCalls}}
		} else {
			ch <- StreamPart{Type: "text-delta", TextDelta: "sunny"}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:    model,
			Prompt:   "weather?",
			StopWhen: []StopCondition{LoopFinished()},
			Tools: map[string]Tool{"weather": {
				Execute: func(_ context.Context, call ToolCall, _ ToolExecutionOptions) (any, error) {
					if call.ToolName != "weather" {
						t.Fatalf("expected repaired tool name, got %q", call.ToolName)
					}
					return "sunny", nil
				},
			}},
			RepairToolCall: func(_ context.Context, opts ToolCallRepairOptions) (*ToolCallPart, error) {
				if !IsNoSuchToolError(opts.Error) {
					t.Fatalf("expected no such tool error, got %v", opts.Error)
				}
				return &ToolCallPart{ToolCallID: opts.ToolCall.ToolCallID, ToolName: "weather", InputRaw: opts.ToolCall.InputRaw}, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}

	var toolCallName string
	var text string
	for part := range result.Stream {
		if part.Type == "tool-call" {
			toolCallName = part.ToolName
		}
		if part.Type == "text-delta" {
			text += part.TextDelta
		}
	}
	if toolCallName != "weather" {
		t.Fatalf("expected repaired tool-call chunk, got %q", toolCallName)
	}
	if text != "sunny" {
		t.Fatalf("expected sunny, got %q", text)
	}
	if len(result.Steps) != 2 || len(result.Steps[0].ToolResults) != 1 {
		t.Fatalf("expected repaired stream tool execution, got %#v", result.Steps)
	}
}

func TestStreamTextAppliesTransformsToCanonicalText(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 4)
		ch <- StreamPart{Type: "text-delta", TextDelta: "Hello"}
		ch <- StreamPart{Type: "text-delta", TextDelta: ", world"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	upper := StreamTransform(func(ctx context.Context, in <-chan StreamPart, _ StreamTransformOptions) <-chan StreamPart {
		out := make(chan StreamPart)
		go func() {
			defer close(out)
			for part := range in {
				if part.Type == "text-delta" {
					part.TextDelta = strings.ToUpper(part.TextDelta)
				}
				select {
				case <-ctx.Done():
					return
				case out <- part:
				}
			}
		}()
		return out
	})

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "say hello",
		},
		Transforms: []StreamTransform{upper},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}

	var streamed string
	for part := range result.Stream {
		if part.Type == "text-delta" {
			streamed += part.TextDelta
		}
	}
	if streamed != "HELLO, WORLD" {
		t.Fatalf("streamed text = %q", streamed)
	}
	if result.Text != "HELLO, WORLD" {
		t.Fatalf("result text = %q", result.Text)
	}
	textPart, ok := result.Response.Messages[0].Content[0].(TextPart)
	if !ok || textPart.Text != "HELLO, WORLD" {
		t.Fatalf("response messages = %#v", result.Response.Messages)
	}
}

func TestStreamTextPreservesTextDeltaProviderMetadata(t *testing.T) {
	metadata := ProviderMetadata{"googleVertex": map[string]any{"thoughtSignature": "sig-1"}}
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 4)
		ch <- StreamPart{Type: "text-delta", TextDelta: "Hello", ProviderMetadata: metadata}
		ch <- StreamPart{Type: "text-delta", TextDelta: ", world"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "hello",
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	for range result.Stream {
	}
	if len(result.Content) != 1 {
		t.Fatalf("expected one content part, got %#v", result.Content)
	}
	text, ok := result.Content[0].(TextPart)
	if !ok {
		t.Fatalf("expected text part, got %#v", result.Content[0])
	}
	if text.Text != "Hello, world" {
		t.Fatalf("unexpected text: %q", text.Text)
	}
	namespace, ok := text.ProviderMetadata["googleVertex"].(map[string]any)
	if !ok || namespace["thoughtSignature"] != "sig-1" {
		t.Fatalf("unexpected text metadata: %#v", text.ProviderMetadata)
	}
}

func TestStreamTextEmitsAbortOnContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	model := NewMockLanguageModel("stream-abort")
	model.StreamFunc = func(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart)
		go func() {
			defer close(ch)
			ch <- StreamPart{Type: "text-delta", TextDelta: "hello"}
			<-ctx.Done()
		}()
		return &LanguageModelStreamResult{Stream: ch}, nil
	}

	result, err := StreamText(ctx, StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{Model: model, Prompt: "hi"},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	var sawAbort bool
	for part := range result.Stream {
		if part.Type == "text-delta" {
			cancel()
		}
		if part.Type == "abort" {
			sawAbort = true
			if part.AbortReason == "" {
				t.Fatalf("expected abort reason")
			}
		}
	}
	if !sawAbort || !result.Aborted {
		t.Fatalf("expected aborted stream, sawAbort=%v result=%#v", sawAbort, result)
	}
}

func TestStreamTextRawChunksRespectIncludeRawChunks(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 3)
		ch <- StreamPart{Type: "raw", Raw: map[string]any{"event": "chunk"}}
		ch <- StreamPart{Type: "text-delta", TextDelta: "ok"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	withoutRaw, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{Model: model, Prompt: "hi"},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	for part := range withoutRaw.Stream {
		if part.Type == "raw" {
			t.Fatalf("did not expect raw part without IncludeRawChunks")
		}
	}

	withRaw, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{Model: model, Prompt: "hi"},
		IncludeRawChunks:    true,
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	var sawRaw bool
	for part := range withRaw.Stream {
		if part.Type == "raw" {
			sawRaw = true
		}
	}
	if !sawRaw {
		t.Fatalf("expected raw part with IncludeRawChunks")
	}
}

func TestStreamTextTransformsToolCallInputBeforeParsing(t *testing.T) {
	calls := 0
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		calls++
		ch := make(chan StreamPart, 4)
		if calls == 1 {
			ch <- StreamPart{Type: "tool-call", ToolCallID: "call-1", ToolName: "weather", ToolInput: `{"city":"nyc"}`}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishToolCalls}}
		} else {
			ch <- StreamPart{Type: "text-delta", TextDelta: "done"}
			ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	normalizeCity := StreamTransform(func(ctx context.Context, in <-chan StreamPart, _ StreamTransformOptions) <-chan StreamPart {
		out := make(chan StreamPart)
		go func() {
			defer close(out)
			for part := range in {
				if part.Type == "tool-call" {
					part.ToolInput = strings.ReplaceAll(part.ToolInput, `"nyc"`, `"NYC"`)
				}
				select {
				case <-ctx.Done():
					return
				case out <- part:
				}
			}
		}()
		return out
	})

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:    model,
			Prompt:   "weather?",
			StopWhen: []StopCondition{LoopFinished()},
			Tools: map[string]Tool{"weather": {
				ValidateInput: func(input any) error {
					if input.(map[string]any)["city"] != "NYC" {
						t.Fatalf("expected transformed input, got %#v", input)
					}
					return nil
				},
				Execute: func(_ context.Context, call ToolCall, _ ToolExecutionOptions) (any, error) {
					if call.Input.(map[string]any)["city"] != "NYC" {
						t.Fatalf("expected transformed execution input, got %#v", call.Input)
					}
					return "ok", nil
				},
			}},
		},
		Transforms: []StreamTransform{normalizeCity},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	for range result.Stream {
	}
	if result.Steps[0].ToolCalls[0].Input.(map[string]any)["city"] != "NYC" {
		t.Fatalf("tool calls = %#v", result.Steps[0].ToolCalls)
	}
}

func TestSmoothStreamChunksTextByWord(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 4)
		ch <- StreamPart{Type: "text-delta", ID: "text-1", TextDelta: "Hello, world!"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	noDelay := time.Duration(0)
	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "say hello",
		},
		Transforms: []StreamTransform{SmoothStream(SmoothStreamOptions{Delay: &noDelay})},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}

	var chunks []string
	for part := range result.Stream {
		if part.Type == "text-delta" {
			chunks = append(chunks, part.TextDelta)
		}
	}
	if len(chunks) != 2 || chunks[0] != "Hello, " || chunks[1] != "world!" {
		t.Fatalf("chunks = %#v", chunks)
	}
	if result.Text != "Hello, world!" {
		t.Fatalf("result text = %q", result.Text)
	}
}
