package ai

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

func TestToolLoopAgentGenerateUsesSettingsAndDefaults(t *testing.T) {
	model := NewMockLanguageModel("agent-model")
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		ID:           "assistant",
		Instructions: "be concise",
		Model:        model,
		Tools: map[string]Tool{
			"weather": {Description: "weather"},
			"search":  {Description: "search"},
		},
		ActiveTools: []string{"search"},
	})

	if agent.Version() != AgentVersion {
		t.Fatalf("version = %q", agent.Version())
	}
	if agent.ID() != "assistant" {
		t.Fatalf("id = %q", agent.ID())
	}

	result, err := agent.Generate(context.Background(), AgentCallOptions{Prompt: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "ok" {
		t.Fatalf("text = %q", result.Text)
	}

	if len(model.GenerateCalls) != 1 {
		t.Fatalf("expected one model call, got %d", len(model.GenerateCalls))
	}
	call := model.GenerateCalls[0]
	if call.Prompt[0].Role != RoleSystem || call.Prompt[0].Text != "be concise" {
		t.Fatalf("expected system instructions in prompt, got %#v", call.Prompt)
	}
	if len(call.Tools) != 1 || call.Tools[0].Name != "search" {
		t.Fatalf("expected only active search tool, got %#v", call.Tools)
	}
}

func TestToolLoopAgentPrepareCallCanOverridePrompt(t *testing.T) {
	model := NewMockLanguageModel("agent-model")
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		Model: model,
		PrepareCall: func(opts AgentPrepareCallOptions) (*AgentPreparedCall, error) {
			prompt := "prepared: " + opts.Call.Prompt
			reasoning := "trace"
			return &AgentPreparedCall{
				Prompt:    &prompt,
				Reasoning: &reasoning,
				Headers:   map[string]string{"X-Prepared": "true"},
			}, nil
		},
	})

	if _, err := agent.Generate(context.Background(), AgentCallOptions{
		Prompt:  "input",
		Headers: map[string]string{"X-Call": "true"},
	}); err != nil {
		t.Fatal(err)
	}

	call := model.GenerateCalls[0]
	if len(call.Prompt) != 1 || TextFromParts(call.Prompt[0].Content) != "prepared: input" {
		t.Fatalf("unexpected prompt: %#v", call.Prompt)
	}
	if call.Reasoning != "trace" {
		t.Fatalf("reasoning = %q", call.Reasoning)
	}
	if call.Headers["X-Call"] != "true" || call.Headers["X-Prepared"] != "true" {
		t.Fatalf("headers = %#v", call.Headers)
	}
}

func TestToolLoopAgentMergesCallbacks(t *testing.T) {
	model := NewMockLanguageModel("agent-model")
	var events []string
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		Model: model,
		OnStart: func(StartEvent) {
			events = append(events, "settings-start")
		},
		OnFinish: func(FinishEvent) {
			events = append(events, "settings-finish")
		},
	})

	_, err := agent.Generate(context.Background(), AgentCallOptions{
		Prompt: "hello",
		OnStart: func(StartEvent) {
			events = append(events, "call-start")
		},
		OnFinish: func(FinishEvent) {
			events = append(events, "call-finish")
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	want := []string{"settings-start", "call-start", "settings-finish", "call-finish"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %#v, want %#v", events, want)
	}
}

func TestToolLoopAgentDefaultStepLimitIsTwenty(t *testing.T) {
	calls := 0
	model := NewMockLanguageModel("loop")
	model.GenerateFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		calls++
		input, _ := json.Marshal(map[string]any{"n": calls})
		return &LanguageModelGenerateResult{
			Content: []Part{ToolCallPart{
				ToolCallID: "call",
				ToolName:   "loop",
				InputRaw:   string(input),
			}},
			FinishReason: FinishReason{Unified: FinishToolCalls, Raw: FinishToolCalls},
		}, nil
	}
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		Model: model,
		Tools: map[string]Tool{
			"loop": {
				Execute: func(context.Context, ToolCall, ToolExecutionOptions) (any, error) {
					return ToolResultOutput{Type: "text", Value: "again"}, nil
				},
			},
		},
	})

	result, err := agent.Generate(context.Background(), AgentCallOptions{Prompt: "loop"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Steps) != 20 {
		t.Fatalf("steps = %d, want 20", len(result.Steps))
	}
}

func TestToolLoopAgentStreamForwardsTransforms(t *testing.T) {
	model := NewMockLanguageModel("agent-stream")
	model.StreamFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 2)
		ch <- StreamPart{Type: "text-delta", TextDelta: "raw"}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}
	agent := NewToolLoopAgent(ToolLoopAgentSettings{
		Model: model,
		Transforms: []StreamTransform{
			func(ctx context.Context, in <-chan StreamPart, opts StreamTransformOptions) <-chan StreamPart {
				out := make(chan StreamPart)
				go func() {
					defer close(out)
					for part := range in {
						if part.Type == "text-delta" {
							part.TextDelta = "transformed"
						}
						out <- part
					}
				}()
				return out
			},
		},
	})

	result, err := agent.Stream(context.Background(), AgentStreamOptions{AgentCallOptions: AgentCallOptions{Prompt: "hello"}})
	if err != nil {
		t.Fatal(err)
	}
	for range result.Stream {
	}
	if result.Text != "transformed" {
		t.Fatalf("expected transformed text, got %q", result.Text)
	}
}

func TestToolLoopAgentGenerateForwardsOutput(t *testing.T) {
	model := NewMockLanguageModel("agent-output")
	model.GenerateFunc = func(ctx context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if opts.ResponseFormat == nil || opts.ResponseFormat.Type != "json" {
			t.Fatalf("expected output response format, got %#v", opts.ResponseFormat)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"result":"yes"}`}},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}
	agent := NewToolLoopAgent(ToolLoopAgentSettings{Model: model})
	result, err := agent.Generate(context.Background(), AgentCallOptions{
		Prompt: "choose",
		Output: ChoiceOutput([]string{"yes", "no"}),
	})
	if err != nil {
		t.Fatal(err)
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatal(err)
	}
	if output != "yes" {
		t.Fatalf("output = %#v", output)
	}
}
