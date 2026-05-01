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
