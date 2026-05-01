package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestGenerateTextRequiresPromptOrMessages(t *testing.T) {
	_, err := GenerateText(context.Background(), GenerateTextOptions{Model: mockModel{}})
	if !errors.Is(err, ErrInvalidPrompt) {
		t.Fatalf("expected invalid prompt error, got %v", err)
	}
}

func TestGenerateTextRejectsSystemInMessagesByDefault(t *testing.T) {
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:    mockModel{},
		Messages: []Message{SystemMessage("nope"), UserMessage("hello")},
	})
	if !errors.Is(err, ErrInvalidPrompt) {
		t.Fatalf("expected invalid prompt error, got %v", err)
	}
}

func TestGenerateTextRunsToolLoop(t *testing.T) {
	calls := 0
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		calls++
		if calls == 1 {
			return &LanguageModelGenerateResult{
				Content: []Part{
					ToolCallPart{ToolCallID: "call-1", ToolName: "weather", InputRaw: `{"city":"NYC"}`},
				},
				FinishReason: FinishReason{Unified: FinishToolCalls, Raw: "tool_use"},
				Usage:        usage(1, 1),
			}, nil
		}
		if len(opts.Prompt) == 0 || opts.Prompt[len(opts.Prompt)-1].Role != RoleTool {
			t.Fatalf("second call should include tool result, got %#v", opts.Prompt)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: "sunny"}},
			FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"},
			Usage:        usage(2, 3),
		}, nil
	}}
	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:    model,
		Prompt:   "weather?",
		StopWhen: []StopCondition{LoopFinished()},
		Tools: map[string]Tool{
			"weather": {
				Description: "Get weather",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
				ValidateInput: func(input any) error {
					m, ok := input.(map[string]any)
					if !ok || m["city"] != "NYC" {
						return errors.New("bad input")
					}
					return nil
				},
				Execute: func(_ context.Context, call ToolCall, _ ToolExecutionOptions) (any, error) {
					b, _ := json.Marshal(call.Input)
					if string(b) != `{"city":"NYC"}` {
						t.Fatalf("unexpected tool input: %s", string(b))
					}
					return "sunny", nil
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	if result.Text != "sunny" {
		t.Fatalf("expected final text sunny, got %q", result.Text)
	}
	if len(result.Steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(result.Steps))
	}
	if len(result.Steps[0].ToolResults) != 1 {
		t.Fatalf("expected first step tool result")
	}
	if result.Usage.TotalTokens == nil || *result.Usage.TotalTokens != 7 {
		t.Fatalf("expected total usage 7, got %#v", result.Usage.TotalTokens)
	}
}

func TestPrepareToolsHonorsNamedToolChoice(t *testing.T) {
	tools := prepareModelTools(map[string]Tool{
		"a": {Description: "A"},
		"b": {Description: "B"},
	}, ToolChoiceFor("b"))
	if len(tools) != 1 || tools[0].Name != "b" {
		t.Fatalf("expected only named tool b, got %#v", tools)
	}
}

func TestGenerateTextFiltersActiveTools(t *testing.T) {
	model := NewMockLanguageModel("lm")
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:       model,
		Prompt:      "hi",
		ActiveTools: []string{"search"},
		Tools: map[string]Tool{
			"weather": {Description: "weather"},
			"search":  {Description: "search"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(model.GenerateCalls[0].Tools) != 1 || model.GenerateCalls[0].Tools[0].Name != "search" {
		t.Fatalf("tools = %#v", model.GenerateCalls[0].Tools)
	}
}

func TestGenerateTextToolExecutionCallbacks(t *testing.T) {
	calls := 0
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		calls++
		if calls == 1 {
			return &LanguageModelGenerateResult{
				Content:      []Part{ToolCallPart{ToolCallID: "call-1", ToolName: "tool", InputRaw: `{}`}},
				FinishReason: FinishReason{Unified: FinishToolCalls},
			}, nil
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: "done"}},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}}
	var events []string
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:    model,
		Prompt:   "run",
		StopWhen: []StopCondition{LoopFinished()},
		Tools: map[string]Tool{"tool": {
			Execute: func(context.Context, ToolCall, ToolExecutionOptions) (any, error) {
				events = append(events, "execute")
				return "ok", nil
			},
		}},
		OnToolExecutionStart: func(event ToolExecutionStartEvent) {
			if event.ToolCall.ToolCallID != "call-1" {
				t.Fatalf("unexpected start event: %#v", event)
			}
			events = append(events, "start")
		},
		OnToolExecutionEnd: func(event ToolExecutionEndEvent) {
			if event.Result.ToolCallID != "call-1" || event.Err != nil {
				t.Fatalf("unexpected end event: %#v", event)
			}
			events = append(events, "end")
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"start", "execute", "end"}
	if len(events) != len(want) {
		t.Fatalf("events = %#v, want %#v", events, want)
	}
	for i := range want {
		if events[i] != want[i] {
			t.Fatalf("events = %#v, want %#v", events, want)
		}
	}
}

func TestGenerateTextRepairsInvalidToolCall(t *testing.T) {
	calls := 0
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		calls++
		if calls == 1 {
			return &LanguageModelGenerateResult{
				Content:      []Part{ToolCallPart{ToolCallID: "call-1", ToolName: "weather", InputRaw: `{"city":`}},
				FinishReason: FinishReason{Unified: FinishToolCalls},
			}, nil
		}
		if len(opts.Prompt) == 0 || opts.Prompt[len(opts.Prompt)-1].Role != RoleTool {
			t.Fatalf("second call should include tool result, got %#v", opts.Prompt)
		}
		result, ok := opts.Prompt[len(opts.Prompt)-1].Content[0].(ToolResultPart)
		if !ok {
			t.Fatalf("expected tool result in second prompt, got %#v", opts.Prompt[len(opts.Prompt)-1].Content)
		}
		input, ok := result.Input.(map[string]any)
		if !ok || input["city"] != "NYC" {
			t.Fatalf("expected repaired input in tool result, got %#v", result.Input)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: "done"}},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}}

	var repairErr error
	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:    model,
		Prompt:   "weather?",
		StopWhen: []StopCondition{LoopFinished()},
		Tools: map[string]Tool{"weather": {
			InputSchema: map[string]any{"type": "object"},
			Execute: func(_ context.Context, call ToolCall, _ ToolExecutionOptions) (any, error) {
				input := call.Input.(map[string]any)
				if input["city"] != "NYC" {
					t.Fatalf("expected repaired tool input, got %#v", call.Input)
				}
				return "sunny", nil
			},
		}},
		RepairToolCall: func(_ context.Context, opts ToolCallRepairOptions) (*ToolCallPart, error) {
			repairErr = opts.Error
			if !errors.Is(opts.Error, ErrInvalidToolInput) {
				t.Fatalf("expected invalid tool input error, got %v", opts.Error)
			}
			if _, ok := opts.InputSchema("weather"); !ok {
				t.Fatalf("expected input schema lookup for weather")
			}
			return &ToolCallPart{ToolCallID: opts.ToolCall.ToolCallID, ToolName: "weather", InputRaw: `{"city":"NYC"}`}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "done" {
		t.Fatalf("expected done, got %q", result.Text)
	}
	if repairErr == nil {
		t.Fatalf("expected repair callback to receive original error")
	}
	if len(result.Steps) != 2 || len(result.Steps[0].ToolResults) != 1 {
		t.Fatalf("expected repaired tool execution, got %#v", result.Steps)
	}
}

func TestGenerateTextWrapsRepairFailure(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content:      []Part{ToolCallPart{ToolCallID: "call-1", ToolName: "weather", InputRaw: `{"city":`}},
			FinishReason: FinishReason{Unified: FinishToolCalls},
		}, nil
	}}

	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "weather?",
		Tools:  map[string]Tool{"weather": {}},
		RepairToolCall: func(context.Context, ToolCallRepairOptions) (*ToolCallPart, error) {
			return nil, errors.New("repair failed")
		},
	})
	if !errors.Is(err, ErrToolCallRepair) {
		t.Fatalf("expected tool call repair error, got %v", err)
	}
}

func TestGenerateTextUsesCustomDownloadForUnsupportedFileURL(t *testing.T) {
	called := false
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if len(opts.Prompt) != 1 || len(opts.Prompt[0].Content) != 1 {
			t.Fatalf("unexpected prompt: %#v", opts.Prompt)
		}
		file, ok := opts.Prompt[0].Content[0].(FilePart)
		if !ok {
			t.Fatalf("expected file part, got %#v", opts.Prompt[0].Content[0])
		}
		if file.Data.Type != FileDataTypeBytes || string(file.Data.Data) != "downloaded" || file.MediaType != "text/plain" {
			t.Fatalf("expected downloaded file part, got %#v", file)
		}
		return &LanguageModelGenerateResult{Content: []Part{TextPart{Text: "ok"}}, FinishReason: FinishReason{Unified: FinishStop}}, nil
	}}
	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model: model,
		Messages: []Message{{
			Role: RoleUser,
			Content: []Part{FilePart{
				Data:      FileData{Type: FileDataTypeURL, URL: "https://assets.example.com/file.txt"},
				MediaType: "text/plain",
			}},
		}},
		Download: func(_ context.Context, rawURL string) ([]byte, string, error) {
			called = true
			if rawURL != "https://assets.example.com/file.txt" {
				t.Fatalf("unexpected download URL: %q", rawURL)
			}
			return []byte("downloaded"), "text/plain", nil
		},
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	if !called {
		t.Fatalf("expected custom download to be called")
	}
}

type mockModel struct{}

func (mockModel) Provider() string { return "mock" }
func (mockModel) ModelID() string  { return "mock-model" }
func (mockModel) SupportedURLs(context.Context) (map[string][]string, error) {
	return nil, nil
}
func (mockModel) DoGenerate(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
	return &LanguageModelGenerateResult{Content: []Part{TextPart{Text: "ok"}}, FinishReason: FinishReason{Unified: FinishStop}}, nil
}
func (mockModel) DoStream(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
	ch := make(chan StreamPart)
	close(ch)
	return &LanguageModelStreamResult{Stream: ch}, nil
}

type sequenceModel struct {
	generate func(LanguageModelCallOptions) (*LanguageModelGenerateResult, error)
	stream   func(LanguageModelCallOptions) (*LanguageModelStreamResult, error)
}

func (m *sequenceModel) Provider() string { return "mock" }
func (m *sequenceModel) ModelID() string  { return "mock-model" }
func (m *sequenceModel) SupportedURLs(context.Context) (map[string][]string, error) {
	return nil, nil
}
func (m *sequenceModel) DoGenerate(_ context.Context, opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
	return m.generate(opts)
}
func (m *sequenceModel) DoStream(_ context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
	if m.stream != nil {
		return m.stream(opts)
	}
	ch := make(chan StreamPart)
	close(ch)
	return &LanguageModelStreamResult{Stream: ch}, nil
}

func usage(input, output int) Usage {
	total := input + output
	return Usage{InputTokens: &input, OutputTokens: &output, TotalTokens: &total}
}
