package ai

import (
	"context"
	"errors"
	"testing"
)

func TestGenerateTextJSONOutputParsesAndSetsResponseFormat(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if opts.ResponseFormat == nil || opts.ResponseFormat.Type != "json" {
			t.Fatalf("expected json response format, got %#v", opts.ResponseFormat)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":"Ada","count":2}`}},
			FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"},
		}, nil
	}}

	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "json",
		Output: JSONOutput(),
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatalf("GetOutput failed: %v", err)
	}
	object := output.(map[string]any)
	if object["name"] != "Ada" {
		t.Fatalf("unexpected output: %#v", output)
	}

	type generated struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}
	typed, err := OutputAs[generated](result)
	if err != nil {
		t.Fatalf("OutputAs failed: %v", err)
	}
	if typed.Name != "Ada" || typed.Count != 2 {
		t.Fatalf("unexpected typed output: %#v", typed)
	}
}

func TestGenerateTextObjectOutputValidatesSchema(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		if opts.ResponseFormat == nil || opts.ResponseFormat.Type != "json" || opts.ResponseFormat.Schema == nil {
			t.Fatalf("expected schema response format, got %#v", opts.ResponseFormat)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":42}`}},
			FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"},
			Response:     ResponseMetadata{ID: "response-1"},
			Usage:        usage(1, 2),
		}, nil
	}}

	_, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "json",
		Output: ObjectOutput(map[string]any{
			"type":     "object",
			"required": []any{"name"},
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		}),
	})
	if !IsNoObjectGeneratedError(err) {
		t.Fatalf("expected NoObjectGeneratedError, got %T %v", err, err)
	}
	var noObject *NoObjectGeneratedError
	if !errors.As(err, &noObject) || noObject.Response.ID != "response-1" || noObject.Cause == nil {
		t.Fatalf("expected no-object context, got %#v", noObject)
	}
}

func TestGenerateTextArrayOutputWrapsSchemaAndUnwrapsElements(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		schema := opts.ResponseFormat.Schema.(map[string]any)
		properties := schema["properties"].(map[string]any)
		if _, ok := properties["elements"]; !ok {
			t.Fatalf("expected wrapped elements schema, got %#v", schema)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"elements":["Ada","Grace"]}`}},
			FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"},
		}, nil
	}}

	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "json",
		Output: ArrayOutput(map[string]any{"type": "string"}),
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatalf("GetOutput failed: %v", err)
	}
	values := output.([]any)
	if len(values) != 2 || values[0] != "Ada" || values[1] != "Grace" {
		t.Fatalf("unexpected array output: %#v", output)
	}
}

func TestGenerateTextChoiceOutputUnwrapsResult(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		schema := opts.ResponseFormat.Schema.(map[string]any)
		properties := schema["properties"].(map[string]any)
		if _, ok := properties["result"]; !ok {
			t.Fatalf("expected result schema, got %#v", schema)
		}
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"result":"sunny"}`}},
			FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"},
		}, nil
	}}

	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "json",
		Output: ChoiceOutput([]string{"sunny", "rainy"}),
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatalf("GetOutput failed: %v", err)
	}
	if output != "sunny" {
		t.Fatalf("unexpected choice output: %#v", output)
	}
}

func TestGenerateTextOutputUnavailableWhenFinishReasonIsNotStop(t *testing.T) {
	model := &sequenceModel{generate: func(opts LanguageModelCallOptions) (*LanguageModelGenerateResult, error) {
		return &LanguageModelGenerateResult{
			Content:      []Part{TextPart{Text: `{"name":"Ada"}`}},
			FinishReason: FinishReason{Unified: FinishLength, Raw: "length"},
		}, nil
	}}

	result, err := GenerateText(context.Background(), GenerateTextOptions{
		Model:  model,
		Prompt: "json",
		Output: JSONOutput(),
	})
	if err != nil {
		t.Fatalf("GenerateText failed: %v", err)
	}
	if _, err := result.GetOutput(); !IsNoOutputGeneratedError(err) {
		t.Fatalf("expected NoOutputGeneratedError, got %T %v", err, err)
	}
}

func TestStreamTextJSONOutputEmitsPartialOutputAndParsesFinal(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		if opts.ResponseFormat == nil || opts.ResponseFormat.Type != "json" {
			t.Fatalf("expected json response format, got %#v", opts.ResponseFormat)
		}
		ch := make(chan StreamPart, 3)
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"name":"Ada"`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `}`}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "json",
			Output: JSONOutput(),
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}

	var sawPartial bool
	for part := range result.Stream {
		if part.Type == "text-delta" && part.PartialOutput != nil {
			object := part.PartialOutput.(map[string]any)
			if object["name"] == "Ada" {
				sawPartial = true
			}
		}
	}
	if !sawPartial {
		t.Fatalf("expected partial output on text stream")
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatalf("GetOutput failed: %v", err)
	}
	if output.(map[string]any)["name"] != "Ada" {
		t.Fatalf("unexpected final output: %#v", output)
	}
}

func TestStreamTextArrayOutputEmitsPartialArraysAndElements(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 8)
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"elements":[`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"content":"one"},`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"content":"two"}`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `]}`}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "json",
			Output: ArrayOutput(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"content": map[string]any{"type": "string"},
				},
			}),
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}

	elements := []any{}
	partials := []any{}
	for part := range result.Stream {
		if part.PartialOutput != nil {
			partials = append(partials, part.PartialOutput)
		}
		if part.Type == "element" {
			elements = append(elements, part.Element)
		}
	}
	if len(elements) != 2 {
		t.Fatalf("expected two element parts, got %#v", elements)
	}
	if len(partials) == 0 {
		t.Fatalf("expected partial array output")
	}
	first := partials[0].([]any)
	if len(first) != 0 {
		t.Fatalf("expected initial empty partial array, got %#v", first)
	}
	output, err := result.GetOutput()
	if err != nil {
		t.Fatalf("GetOutput failed: %v", err)
	}
	if len(output.([]any)) != 2 {
		t.Fatalf("unexpected final output: %#v", output)
	}
}

func TestStreamTextChoiceOutputEmitsUnambiguousPartial(t *testing.T) {
	model := &sequenceModel{stream: func(opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		ch := make(chan StreamPart, 5)
		ch <- StreamPart{Type: "text-delta", TextDelta: `{"result":"su`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `nny"`}
		ch <- StreamPart{Type: "text-delta", TextDelta: `}`}
		ch <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop, Raw: "stop"}}
		close(ch)
		return &LanguageModelStreamResult{Stream: ch}, nil
	}}

	result, err := StreamText(context.Background(), StreamTextOptions{
		GenerateTextOptions: GenerateTextOptions{
			Model:  model,
			Prompt: "json",
			Output: ChoiceOutput([]string{"sunny", "rainy", "snowy"}),
		},
	})
	if err != nil {
		t.Fatalf("StreamText failed: %v", err)
	}
	var partial any
	for part := range result.Stream {
		if part.PartialOutput != nil {
			partial = part.PartialOutput
		}
	}
	if partial != "sunny" {
		t.Fatalf("expected sunny partial, got %#v", partial)
	}
}
