package ai

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestGenerateObjectParsesJSON(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if opts.ResponseFormat == nil || opts.ResponseFormat.Type != "json" {
			t.Fatalf("expected json response format, got %#v", opts.ResponseFormat)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":"Ada"}`}},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}}
	result, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Prompt: "json",
		Schema: map[string]any{"type": "object"},
	})
	if err != nil {
		t.Fatalf("GenerateObject failed: %v", err)
	}
	object := result.Object.(map[string]any)
	if object["name"] != "Ada" {
		t.Fatalf("unexpected object: %#v", object)
	}
}

func TestGenerateObjectRepairText(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":`}},
			FinishReason: FinishReason{Unified: FinishStop},
		}, nil
	}}
	result, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Prompt: "json",
		Schema: map[string]any{"type": "object"},
		RepairText: func(opts RepairTextOptions) (string, error) {
			if opts.Text == "" || opts.Error == nil {
				return "", errors.New("missing repair context")
			}
			return `{"name":"Grace"}`, nil
		},
	})
	if err != nil {
		t.Fatalf("GenerateObject failed: %v", err)
	}
	if result.Object.(map[string]any)["name"] != "Grace" {
		t.Fatalf("unexpected object: %#v", result.Object)
	}
}

func TestGenerateObjectInvalidJSONUsesNamedError(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":`}},
			FinishReason: FinishReason{Unified: FinishStop},
			Usage:        Usage{TotalTokens: intPtr(4)},
			Response:     ResponseMetadata{ID: "response-1"},
		}, nil
	}}
	_, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Prompt: "json",
		Schema: map[string]any{"type": "object"},
	})
	if !IsNoObjectGeneratedError(err) {
		t.Fatalf("expected NoObjectGeneratedError, got %T %v", err, err)
	}
	var noObject *NoObjectGeneratedError
	if !errors.As(err, &noObject) {
		t.Fatalf("expected typed no-object error, got %T", err)
	}
	if noObject.Text != `{"name":` || noObject.Response.ID != "response-1" {
		t.Fatalf("expected error context to be preserved, got %#v", noObject)
	}
}

func TestGenerateObjectEnumValidationUsesNamedError(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{Content: []Part{TextPart{Text: `"nope"`}}}, nil
	}}
	_, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Output: OutputEnum,
		Enum:   []string{"yes"},
		Prompt: "json",
	})
	if !IsNoObjectGeneratedError(err) {
		t.Fatalf("expected NoObjectGeneratedError, got %T %v", err, err)
	}
}

func TestGenerateObjectSchemaValidationUsesNamedError(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{Content: []Part{TextPart{Text: `{"name":42}`}}}, nil
	}}
	_, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Prompt: "json",
		Schema: map[string]any{
			"type":     "object",
			"required": []any{"name"},
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		},
	})
	if !IsNoObjectGeneratedError(err) {
		t.Fatalf("expected NoObjectGeneratedError, got %T %v", err, err)
	}
	var noObject *NoObjectGeneratedError
	if !errors.As(err, &noObject) || noObject.Cause == nil {
		t.Fatalf("expected schema cause on no-object error, got %#v", noObject)
	}
}

func TestGenerateObjectArrayOutputStrategyUnwrapsElements(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		schema := opts.ResponseFormat.Schema.(map[string]any)
		if schema["type"] != "object" {
			t.Fatalf("expected wrapped object schema, got %#v", schema)
		}
		properties := schema["properties"].(map[string]any)
		if _, ok := properties["elements"]; !ok {
			t.Fatalf("expected elements wrapper schema, got %#v", schema)
		}
		return &LanguageModelGenerateResult{
			Content: []Part{TextPart{Text: `{"elements":["Ada","Grace"]}`}},
		}, nil
	}}
	result, err := GenerateObject(context.Background(), GenerateObjectOptions{
		Model:  model,
		Prompt: "json",
		Output: OutputArray,
		Schema: map[string]any{"type": "string"},
	})
	if err != nil {
		t.Fatalf("GenerateObject failed: %v", err)
	}
	values := result.Object.([]any)
	if len(values) != 2 || values[0] != "Ada" || values[1] != "Grace" {
		t.Fatalf("unexpected array result: %#v", result.Object)
	}
}

func TestInjectJSONInstruction(t *testing.T) {
	got := InjectJSONInstruction(InjectJSONInstructionOptions{
		Prompt: "Return a person.",
		Schema: map[string]any{"type": "object"},
	})
	want := "Return a person.\n\nJSON schema:\n{\"type\":\"object\"}\nYou MUST answer with a JSON object that matches the JSON schema above."
	if got != want {
		t.Fatalf("unexpected instruction:\n%s", got)
	}
	generic := InjectJSONInstruction(InjectJSONInstructionOptions{})
	if generic != "You MUST answer with JSON." {
		t.Fatalf("unexpected generic instruction: %q", generic)
	}
}

func TestObjectInputValidationMatchesOutputModes(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{Content: []Part{TextPart{Text: `{}`}}}, nil
	}}
	cases := []GenerateObjectOptions{
		{Model: model, Prompt: "json", Output: OutputNoSchema, SchemaName: "x"},
		{Model: model, Prompt: "json", Output: OutputEnum, Schema: map[string]any{"type": "string"}, Enum: []string{"a"}},
		{Model: model, Prompt: "json", Output: OutputArray, Schema: map[string]any{"type": "string"}, Enum: []string{"a"}},
	}
	for _, opts := range cases {
		if _, err := GenerateObject(context.Background(), opts); err == nil {
			t.Fatalf("expected validation error for %#v", opts)
		}
	}
}

func TestStreamObjectEmitsPartialObjects(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 3)
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"name":"Ada"`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `}`}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}
	result, err := StreamObject(context.Background(), StreamObjectOptions{
		GenerateObjectOptions: GenerateObjectOptions{
			Model:  model,
			Prompt: "json",
			Schema: map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatalf("StreamObject failed: %v", err)
	}
	var seen bool
	for part := range result.Stream {
		if part.Type == "object" && part.Err == nil {
			data, _ := json.Marshal(part.Object)
			if string(data) == `{"name":"Ada"}` {
				seen = true
			}
		}
	}
	if !seen {
		t.Fatalf("expected parsed object part")
	}
}

func TestStreamObjectEmitsFinalValidationError(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 2)
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"name":42}`}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}
	result, err := StreamObject(context.Background(), StreamObjectOptions{
		GenerateObjectOptions: GenerateObjectOptions{
			Model:  model,
			Prompt: "json",
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("StreamObject failed: %v", err)
	}
	var sawError bool
	for part := range result.Stream {
		if part.Type == "error" && IsNoObjectGeneratedError(part.Err) {
			sawError = true
		}
	}
	if !sawError {
		t.Fatalf("expected final schema validation error")
	}
}
