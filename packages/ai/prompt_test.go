package ai

import (
	"bytes"
	"context"
	"errors"
	"testing"
)

func TestStandardizePromptRejectsInvalidMessageRole(t *testing.T) {
	_, err := standardizePrompt("", "", []Message{{Role: Role("developer"), Text: "debug"}}, false)
	if !errors.Is(err, ErrInvalidMessageRole) || !IsInvalidMessageRoleError(err) {
		t.Fatalf("expected invalid message role error, got %v", err)
	}
}

func TestStandardizePromptRejectsSystemMessageParts(t *testing.T) {
	_, err := standardizePrompt("", "", []Message{{
		Role:    RoleSystem,
		Content: []Part{TextPart{Text: "instructions"}},
	}}, true)
	if !errors.Is(err, ErrInvalidPrompt) {
		t.Fatalf("expected invalid prompt for system message parts, got %v", err)
	}
}

func TestStandardizePromptValidatesFilePartContract(t *testing.T) {
	_, err := standardizePrompt("", "", []Message{{
		Role: RoleUser,
		Content: []Part{FilePart{
			Data: FileData{Type: FileDataTypeURL, URL: "https://example.com/file.pdf"},
		}},
	}}, false)
	if !errors.Is(err, ErrInvalidPrompt) {
		t.Fatalf("expected invalid prompt for missing media type, got %v", err)
	}

	_, err = standardizePrompt("", "", []Message{{
		Role: RoleUser,
		Content: []Part{FilePart{
			Data:      FileData{Type: FileDataTypeReference, ProviderReference: ProviderReference{"openai": "file-123"}},
			MediaType: "application/pdf",
		}},
	}}, false)
	if err != nil {
		t.Fatalf("expected provider-reference file part to validate, got %v", err)
	}
}

func TestStandardizePromptRejectsUnsupportedToolMessagePart(t *testing.T) {
	_, err := standardizePrompt("", "", []Message{{
		Role:    RoleTool,
		Content: []Part{TextPart{Text: "not a tool result"}},
	}}, false)
	if !errors.Is(err, ErrInvalidPrompt) {
		t.Fatalf("expected invalid prompt for unsupported tool message part, got %v", err)
	}
}

func TestToLanguageModelPromptReturnsTypedMissingToolResults(t *testing.T) {
	_, err := toLanguageModelPrompt(standardizedPrompt{Messages: []Message{{
		Role: RoleAssistant,
		Content: []Part{ToolCallPart{
			ToolCallID: "call-1",
			ToolName:   "weather",
			Input:      map[string]any{"city": "SF"},
		}},
	}}}, nil)
	if !errors.Is(err, ErrMissingToolResults) || !IsMissingToolResultsError(err) {
		t.Fatalf("expected typed missing tool results error, got %v", err)
	}
}

func TestStandardizePromptRejectsInvalidToolCallAndResultParts(t *testing.T) {
	cases := []Message{
		{Role: RoleAssistant, Content: []Part{ToolCallPart{ToolName: "weather", InputRaw: `{}`}}},
		{Role: RoleAssistant, Content: []Part{ToolCallPart{ToolCallID: "call-1", InputRaw: `{}`}}},
		{Role: RoleAssistant, Content: []Part{ToolCallPart{ToolCallID: "call-1", ToolName: "weather", InputRaw: `{`}}},
		{Role: RoleAssistant, Content: []Part{ToolResultPart{ToolCallID: "call-1", ToolName: "weather"}}},
		{Role: RoleTool, Content: []Part{ToolResultPart{ToolName: "weather"}}},
	}
	for _, message := range cases {
		if _, err := standardizePrompt("", "", []Message{message}, true); !errors.Is(err, ErrInvalidPrompt) {
			t.Fatalf("expected invalid prompt for %#v, got %v", message, err)
		}
	}
}

func TestToLanguageModelPromptRejectsOrphanToolResult(t *testing.T) {
	_, err := toLanguageModelPrompt(standardizedPrompt{Messages: []Message{{
		Role: RoleTool,
		Content: []Part{ToolResultPart{
			ToolCallID: "call-1",
			ToolName:   "weather",
			Output:     ToolResultOutput{Type: "text", Value: "sunny"},
		}},
	}}}, nil)
	if !errors.Is(err, ErrMissingToolResults) {
		t.Fatalf("expected missing tool result error, got %v", err)
	}
}

func TestToLanguageModelPromptAllowsProviderExecutedAssistantResult(t *testing.T) {
	_, err := toLanguageModelPrompt(standardizedPrompt{Messages: []Message{{
		Role: RoleAssistant,
		Content: []Part{
			ToolCallPart{ToolCallID: "call-1", ToolName: "code_execution", ProviderExecuted: true},
			ToolResultPart{ToolCallID: "call-1", ToolName: "code_execution", ProviderExecuted: true},
		},
	}}}, nil)
	if err != nil {
		t.Fatalf("provider-executed assistant tool result should be allowed: %v", err)
	}
}

func TestToLanguageModelPromptDownloadsUnsupportedURLFileParts(t *testing.T) {
	prompt := standardizedPrompt{Messages: []Message{{
		Role: RoleUser,
		Content: []Part{FilePart{
			Data:      FileData{Type: FileDataTypeURL, URL: "https://assets.example.com/photo"},
			MediaType: "image/jpeg",
			Filename:  "photo.jpg",
		}},
	}}}
	got, err := toLanguageModelPromptWithOptions(context.Background(), prompt, nil, promptConversionOptions{
		SupportedURLs: map[string][]string{
			"image/jpeg": {"https://cdn.example.com/*"},
		},
		Download: func(_ context.Context, rawURL string) ([]byte, string, error) {
			if rawURL != "https://assets.example.com/photo" {
				t.Fatalf("unexpected download URL: %q", rawURL)
			}
			return []byte{0xff, 0xd8, 0xff}, "image/jpeg", nil
		},
	})
	if err != nil {
		t.Fatalf("toLanguageModelPromptWithOptions failed: %v", err)
	}
	part := got[0].Content[0].(FilePart)
	if part.Data.Type != FileDataTypeBytes || !bytes.Equal(part.Data.Data, []byte{0xff, 0xd8, 0xff}) {
		t.Fatalf("expected downloaded bytes, got %#v", part.Data)
	}
	if prompt.Messages[0].Content[0].(FilePart).Data.Type != FileDataTypeURL {
		t.Fatalf("original prompt was mutated")
	}
}

func TestToLanguageModelPromptKeepsSupportedURLFileParts(t *testing.T) {
	prompt := standardizedPrompt{Messages: []Message{{
		Role: RoleUser,
		Content: []Part{FilePart{
			Data:      FileData{Type: FileDataTypeURL, URL: "https://cdn.example.com/photo.jpg"},
			MediaType: "image/jpeg",
		}},
	}}}
	got, err := toLanguageModelPromptWithOptions(context.Background(), prompt, nil, promptConversionOptions{
		SupportedURLs: map[string][]string{
			"image/*": {"https://cdn.example.com/*"},
		},
		Download: func(context.Context, string) ([]byte, string, error) {
			t.Fatalf("supported URL should not be downloaded")
			return nil, "", nil
		},
	})
	if err != nil {
		t.Fatalf("toLanguageModelPromptWithOptions failed: %v", err)
	}
	part := got[0].Content[0].(FilePart)
	if part.Data.Type != FileDataTypeURL || part.Data.URL != "https://cdn.example.com/photo.jpg" {
		t.Fatalf("expected URL to pass through, got %#v", part.Data)
	}
}
