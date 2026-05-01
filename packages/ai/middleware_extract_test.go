package ai

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

func TestExtractJSONMiddlewareGenerateStripsCodeFence(t *testing.T) {
	model := NewMockLanguageModel("lm")
	model.GenerateFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content: []Part{
				TextPart{Text: "```json\n{\"name\":\"Ada\"}\n```"},
				ReasoningPart{Text: "kept"},
			},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}

	result, err := WrapLanguageModel(model, ExtractJSONMiddleware()).DoGenerate(context.Background(), LanguageModelCallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got := result.Content[0].(TextPart).Text; got != `{"name":"Ada"}` {
		t.Fatalf("text = %q", got)
	}
	if got := result.Content[1].(ReasoningPart).Text; got != "kept" {
		t.Fatalf("reasoning = %q", got)
	}
}

func TestExtractReasoningMiddlewareGenerateSplitsTaggedText(t *testing.T) {
	model := NewMockLanguageModel("lm")
	model.GenerateFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content: []Part{TextPart{Text: "before <think>one</think> after <think>two</think> done"}},
		}, nil
	}

	result, err := WrapLanguageModel(model, ExtractReasoningMiddleware(ExtractReasoningMiddlewareOptions{
		TagName:   "think",
		Separator: "\n",
	})).DoGenerate(context.Background(), LanguageModelCallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 2 {
		t.Fatalf("content = %#v", result.Content)
	}
	if got := result.Content[0].(ReasoningPart).Text; got != "one\ntwo" {
		t.Fatalf("reasoning = %q", got)
	}
	if got := result.Content[1].(TextPart).Text; got != "before \n after \n done" {
		t.Fatalf("text = %q", got)
	}
}

func TestSimulateStreamingMiddlewareUsesGenerateResult(t *testing.T) {
	model := NewMockLanguageModel("lm")
	model.GenerateFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content: []Part{
				TextPart{Text: "hello"},
				ReasoningPart{Text: "trace"},
				ToolCallPart{ToolCallID: "call-1", ToolName: "weather", Input: map[string]any{"city": "NYC"}},
			},
			FinishReason: FinishReason{Unified: FinishToolCalls},
			Usage:        Usage{OutputTokens: intPtr(3)},
		}, nil
	}

	result, err := WrapLanguageModel(model, SimulateStreamingMiddleware()).DoStream(context.Background(), LanguageModelCallOptions{})
	if err != nil {
		t.Fatal(err)
	}
	parts := collectStreamParts(result.Stream)
	if got := streamPartTypes(parts); !reflect.DeepEqual(got, []string{
		"stream-start",
		"response-metadata",
		"text-start",
		"text-delta",
		"text-end",
		"reasoning-start",
		"reasoning-delta",
		"reasoning-end",
		"tool-call",
		"finish",
	}) {
		t.Fatalf("types = %#v", got)
	}
	if !strings.Contains(parts[8].ToolInput, `"city":"NYC"`) {
		t.Fatalf("tool input = %q", parts[8].ToolInput)
	}
	if parts[9].FinishReason.Unified != FinishToolCalls || parts[9].Usage.OutputTokens == nil || *parts[9].Usage.OutputTokens != 3 {
		t.Fatalf("finish = %#v", parts[9])
	}
}

func TestAddToolInputExamplesMiddlewareTransformsCallOptions(t *testing.T) {
	model := NewMockLanguageModel("lm")
	wrapped := WrapLanguageModel(model, AddToolInputExamplesMiddleware())

	_, err := wrapped.DoGenerate(context.Background(), LanguageModelCallOptions{
		Tools: []ModelTool{{
			Type:          "function",
			Name:          "weather",
			Description:   "Get weather.",
			InputExamples: []any{map[string]any{"input": map[string]any{"city": "NYC"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := model.GenerateCalls[0].Tools[0]
	if !strings.Contains(got.Description, "Get weather.\n\nInput Examples:\n") || !strings.Contains(got.Description, `"city":"NYC"`) {
		t.Fatalf("description = %q", got.Description)
	}
	if got.InputExamples != nil {
		t.Fatalf("input examples were not removed: %#v", got.InputExamples)
	}
}

func collectStreamParts(stream <-chan StreamPart) []StreamPart {
	var parts []StreamPart
	for part := range stream {
		parts = append(parts, part)
	}
	return parts
}

func streamPartTypes(parts []StreamPart) []string {
	types := make([]string, len(parts))
	for i, part := range parts {
		types[i] = part.Type
	}
	return types
}
