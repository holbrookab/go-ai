package ai

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestUIMessageTypeGuards(t *testing.T) {
	if !IsTextUIPart(UIPart{Type: "text"}) {
		t.Fatalf("expected text part")
	}
	if !IsDataUIPart(UIPart{Type: "data-weather"}) {
		t.Fatalf("expected data part")
	}
	if !IsToolUIPart(UIPart{Type: "tool-search"}) {
		t.Fatalf("expected static tool part")
	}
	if ToolName(UIPart{Type: "tool-get-weather"}) != "get-weather" {
		t.Fatalf("unexpected static tool name")
	}
	if ToolName(UIPart{Type: "dynamic-tool", ToolName: "lookup"}) != "lookup" {
		t.Fatalf("unexpected dynamic tool name")
	}
}

func TestValidateUIMessagesRejectsInvalidMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []UIMessage
	}{
		{name: "empty"},
		{
			name: "missing id",
			messages: []UIMessage{{
				Role:  RoleUser,
				Parts: []UIPart{{Type: "text", Text: "hi"}},
			}},
		},
		{
			name: "empty parts",
			messages: []UIMessage{{
				ID:   "1",
				Role: RoleUser,
			}},
		},
		{
			name: "tool missing call id",
			messages: []UIMessage{{
				ID:    "1",
				Role:  RoleAssistant,
				Parts: []UIPart{{Type: "tool-weather", State: "input-available", Input: map[string]any{"city": "NYC"}}},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateUIMessages(tt.messages)
			if !errors.Is(err, ErrInvalidUIMessage) {
				t.Fatalf("expected invalid UI message error, got %v", err)
			}
		})
	}
}

func TestValidateUIMessagesValidatesMetadataDataAndToolSchemas(t *testing.T) {
	messages := []UIMessage{{
		ID:       "assistant",
		Role:     RoleAssistant,
		Metadata: map[string]any{"tenant": "acme"},
		Parts: []UIPart{
			{Type: "data-weather", ID: "data-1", Data: map[string]any{"city": "NYC"}},
			{Type: "tool-weather", ToolCallID: "call-1", State: "input-available", Input: map[string]any{"city": "NYC"}},
		},
	}}

	err := ValidateUIMessages(messages, ValidateUIMessagesOptions{
		MetadataSchema: map[string]any{
			"type":       "object",
			"required":   []any{"tenant"},
			"properties": map[string]any{"tenant": map[string]any{"type": "string"}},
		},
		DataSchemas: map[string]any{
			"weather": map[string]any{
				"type":       "object",
				"required":   []any{"city"},
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
		},
		Tools: map[string]Tool{
			"weather": {
				InputSchema: map[string]any{
					"type":       "object",
					"required":   []any{"city"},
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ValidateUIMessages failed: %v", err)
	}
}

func TestValidateUIMessagesRejectsInvalidMetadataSchema(t *testing.T) {
	err := ValidateUIMessages([]UIMessage{{
		ID:       "1",
		Role:     RoleUser,
		Metadata: map[string]any{"tenant": 123},
		Parts:    []UIPart{{Type: "text", Text: "hi"}},
	}}, ValidateUIMessagesOptions{
		MetadataSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"tenant": map[string]any{"type": "string"}},
		},
	})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid UI message error, got %v", err)
	}
	if !strings.Contains(err.Error(), `messages[0].metadata (id: "1") validation failed: no object generated: $.tenant must be string`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateUIMessagesRejectsDataWithoutSchema(t *testing.T) {
	err := ValidateUIMessages([]UIMessage{{
		ID:    "1",
		Role:  RoleAssistant,
		Parts: []UIPart{{Type: "data-weather", ID: "data-1", Data: map[string]any{"city": "NYC"}}},
	}}, ValidateUIMessagesOptions{DataSchemas: map[string]any{}})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid UI message error, got %v", err)
	}
	if !strings.Contains(err.Error(), `messages[0].parts[0].data (id: "data-1", name: "weather") validation failed: no data schema found`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateUIMessagesRejectsInvalidToolInputSchema(t *testing.T) {
	err := ValidateUIMessages([]UIMessage{{
		ID:   "1",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "tool-weather",
			ToolCallID: "call-1",
			State:      "input-available",
			Input:      map[string]any{"city": 123},
		}},
	}}, ValidateUIMessagesOptions{
		Tools: map[string]Tool{
			"weather": {
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
				},
			},
		},
	})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid UI message error, got %v", err)
	}
	if !strings.Contains(err.Error(), `messages[0].parts[0].input (toolCallId: "call-1", toolName: "weather") validation failed: no object generated: $.city must be string`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateUIMessagesRejectsInvalidToolOutputSchema(t *testing.T) {
	err := ValidateUIMessages([]UIMessage{{
		ID:   "1",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "tool-weather",
			ToolCallID: "call-1",
			State:      "output-available",
			Input:      map[string]any{"city": "NYC"},
			Output:     map[string]any{"forecast": 123},
		}},
	}}, ValidateUIMessagesOptions{
		Tools: map[string]Tool{
			"weather": {
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
				},
				OutputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"forecast": map[string]any{"type": "string"}},
				},
			},
		},
	})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid UI message error, got %v", err)
	}
	if !strings.Contains(err.Error(), `messages[0].parts[0].output (toolCallId: "call-1", toolName: "weather") validation failed: no object generated: $.forecast must be string`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateUIMessagesRejectsInvalidDynamicToolSchemas(t *testing.T) {
	tools := map[string]Tool{
		"weather": {
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
			},
			OutputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"forecast": map[string]any{"type": "string"}},
			},
		},
	}
	err := ValidateUIMessages([]UIMessage{{
		ID:   "1",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "dynamic-tool",
			ToolName:   "weather",
			ToolCallID: "call-1",
			State:      "input-available",
			Input:      map[string]any{"city": 123},
			Dynamic:    true,
		}},
	}}, ValidateUIMessagesOptions{Tools: tools})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid dynamic tool input, got %v", err)
	}

	err = ValidateUIMessages([]UIMessage{{
		ID:   "1",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "dynamic-tool",
			ToolName:   "weather",
			ToolCallID: "call-1",
			State:      "output-available",
			Input:      map[string]any{"city": "NYC"},
			Output:     map[string]any{"forecast": 42},
			Dynamic:    true,
		}},
	}}, ValidateUIMessagesOptions{Tools: tools})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid dynamic tool output, got %v", err)
	}
}

func TestConvertToModelMessagesValidatesToolSchemas(t *testing.T) {
	_, err := ConvertToModelMessages([]UIMessage{{
		ID:   "1",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "tool-weather",
			ToolCallID: "call-1",
			State:      "output-available",
			Input:      map[string]any{"city": "NYC"},
			Output:     map[string]any{"forecast": 42},
		}},
	}}, ConvertToModelMessagesOptions{
		Tools: map[string]Tool{
			"weather": {
				OutputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"forecast": map[string]any{"type": "string"}},
				},
			},
		},
	})
	if !errors.Is(err, ErrInvalidUIMessage) {
		t.Fatalf("expected conversion validation error, got %v", err)
	}
}

func TestSafeValidateUIMessagesReturnsErrorResult(t *testing.T) {
	result := SafeValidateUIMessages(nil)
	if result.Success || result.Error == nil {
		t.Fatalf("expected failed validation result, got %#v", result)
	}
	if !errors.Is(result.Error, ErrInvalidUIMessage) {
		t.Fatalf("expected invalid UI message error, got %v", result.Error)
	}
}

func TestConvertToModelMessagesConvertsSystemAndUserParts(t *testing.T) {
	messages := []UIMessage{
		{
			ID:   "sys",
			Role: RoleSystem,
			Parts: []UIPart{
				{Type: "text", Text: "You are "},
				{Type: "text", Text: "helpful.", ProviderMetadata: ProviderMetadata{"mock": map[string]any{"cache": true}}},
			},
		},
		{
			ID:   "user",
			Role: RoleUser,
			Parts: []UIPart{
				{Type: "text", Text: "Look at this"},
				{Type: "file", MediaType: "image/png", Filename: "chart.png", URL: "https://example.test/chart.png"},
				{Type: "data-note", Data: "converted"},
			},
		},
	}

	got, err := ConvertToModelMessages(messages, ConvertToModelMessagesOptions{
		ConvertDataPart: func(part UIPart) (Part, bool, error) {
			return TextPart{Text: part.Data.(string)}, true, nil
		},
	})
	if err != nil {
		t.Fatalf("ConvertToModelMessages failed: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(got))
	}
	if got[0].Role != RoleSystem || got[0].Text != "You are helpful." {
		t.Fatalf("unexpected system message: %#v", got[0])
	}
	if !reflect.DeepEqual(got[0].ProviderOptions, ProviderOptions{"mock": map[string]any{"cache": true}}) {
		t.Fatalf("unexpected provider options: %#v", got[0].ProviderOptions)
	}
	if got[1].Role != RoleUser || len(got[1].Content) != 3 {
		t.Fatalf("unexpected user message: %#v", got[1])
	}
	file, ok := got[1].Content[1].(FilePart)
	if !ok {
		t.Fatalf("expected file part, got %#v", got[1].Content[1])
	}
	if file.Data.Type != "url" || file.Data.URL != "https://example.test/chart.png" || file.MediaType != "image/png" {
		t.Fatalf("unexpected file part: %#v", file)
	}
}

func TestConvertToModelMessagesConvertsAssistantToolBlocks(t *testing.T) {
	messages := []UIMessage{{
		ID:   "assistant",
		Role: RoleAssistant,
		Parts: []UIPart{
			{Type: "reasoning", Text: "Need weather."},
			{Type: "text", Text: "Checking."},
			{
				Type:       "tool-weather",
				ToolCallID: "call-1",
				State:      "output-available",
				Input:      map[string]any{"city": "NYC"},
				Output:     "sunny",
			},
			{Type: "step-start"},
			{Type: "text", Text: "It is sunny."},
		},
	}}

	got, err := ConvertToModelMessages(messages, ConvertToModelMessagesOptions{})
	if err != nil {
		t.Fatalf("ConvertToModelMessages failed: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected assistant/tool/assistant messages, got %#v", got)
	}
	if got[0].Role != RoleAssistant || got[1].Role != RoleTool || got[2].Role != RoleAssistant {
		t.Fatalf("unexpected roles: %#v", got)
	}
	if _, ok := got[0].Content[0].(ReasoningPart); !ok {
		t.Fatalf("expected reasoning part, got %#v", got[0].Content[0])
	}
	call, ok := got[0].Content[2].(ToolCallPart)
	if !ok || call.ToolName != "weather" || call.ToolCallID != "call-1" {
		t.Fatalf("unexpected tool call: %#v", got[0].Content[2])
	}
	result, ok := got[1].Content[0].(ToolResultPart)
	if !ok {
		t.Fatalf("expected tool result, got %#v", got[1].Content[0])
	}
	if result.ToolName != "weather" || result.Output.Type != "text" || result.Output.Value != "sunny" {
		t.Fatalf("unexpected tool result: %#v", result)
	}
}

func TestConvertToModelMessagesCanIgnoreIncompleteToolCalls(t *testing.T) {
	got, err := ConvertToModelMessages([]UIMessage{{
		ID:   "assistant",
		Role: RoleAssistant,
		Parts: []UIPart{
			{Type: "tool-weather", ToolCallID: "call-1", State: "input-streaming"},
			{Type: "text", Text: "still useful"},
		},
	}}, ConvertToModelMessagesOptions{IgnoreIncompleteToolCalls: true})
	if err != nil {
		t.Fatalf("ConvertToModelMessages failed: %v", err)
	}
	if len(got) != 1 || got[0].Role != RoleAssistant || len(got[0].Content) != 1 {
		t.Fatalf("unexpected converted messages: %#v", got)
	}
	text, ok := got[0].Content[0].(TextPart)
	if !ok || text.Text != "still useful" {
		t.Fatalf("unexpected text part: %#v", got[0].Content[0])
	}
}

func TestAppendResponseMessagesAppliesToolResults(t *testing.T) {
	messages := []UIMessage{{
		ID:   "assistant",
		Role: RoleAssistant,
		Parts: []UIPart{{
			Type:       "tool-weather",
			ToolCallID: "call-1",
			State:      "input-available",
			Input:      map[string]any{"city": "NYC"},
		}},
	}}

	got := AppendResponseMessages(messages, []Message{
		ToolMessage(ToolResultPart{
			ToolCallID: "call-1",
			ToolName:   "weather",
			Result:     map[string]any{"forecast": "sunny"},
			Output:     ToolResultOutput{Type: "json", Value: map[string]any{"forecast": "sunny"}},
		}),
	})
	if len(got) != 1 || len(got[0].Parts) != 1 {
		t.Fatalf("unexpected messages: %#v", got)
	}
	if got[0].Parts[0].State != "output-available" {
		t.Fatalf("expected output-available, got %#v", got[0].Parts[0])
	}
	if !reflect.DeepEqual(got[0].Parts[0].Output, map[string]any{"forecast": "sunny"}) {
		t.Fatalf("unexpected output: %#v", got[0].Parts[0].Output)
	}
}
