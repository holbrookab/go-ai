package ai

import (
	"context"
	"reflect"
	"testing"
)

func TestFilterActiveTools(t *testing.T) {
	tools := map[string]Tool{
		"weather": {Description: "weather tool"},
		"search":  {Description: "search tool"},
	}

	got := FilterActiveTools(tools, []string{"search", "missing"})

	if len(got) != 1 {
		t.Fatalf("expected one active tool, got %#v", got)
	}
	if got["search"].Description != "search tool" {
		t.Fatalf("expected search tool to be retained, got %#v", got)
	}
	unfiltered := FilterActiveTools(tools, nil)
	unfiltered["extra"] = Tool{Description: "same map"}
	if _, ok := tools["extra"]; !ok {
		t.Fatalf("nil activeTools should return original tools map")
	}
}

func TestExtractTextAndReasoningContent(t *testing.T) {
	parts := []Part{
		ReasoningPart{Text: "first"},
		TextPart{Text: "hello "},
		FilePart{Filename: "ignored.txt"},
		TextPart{Text: "world"},
		ReasoningPart{Text: "second"},
	}

	text, ok := ExtractTextContent(parts)
	if !ok || text != "hello world" {
		t.Fatalf("unexpected text content: %q, %v", text, ok)
	}

	reasoning, ok := ExtractReasoningContent(parts)
	if !ok || reasoning != "first\nsecond" {
		t.Fatalf("unexpected reasoning content: %q, %v", reasoning, ok)
	}

	if text, ok := ExtractTextContent([]Part{FilePart{Filename: "only-file"}}); ok || text != "" {
		t.Fatalf("expected no text content, got %q, %v", text, ok)
	}
}

func TestPruneMessagesRemovesReasoningAndToolCalls(t *testing.T) {
	messages := []Message{
		UserMessage("question"),
		{
			Role: RoleAssistant,
			Content: []Part{
				ReasoningPart{Text: "hidden"},
				TextPart{Text: "I'll check."},
				ToolCallPart{ToolCallID: "call-1", ToolName: "weather"},
			},
		},
		ToolMessage(ToolResultPart{ToolCallID: "call-1", ToolName: "weather", Output: ToolResultOutput{Type: "json", Value: map[string]any{"ok": true}}}),
		{
			Role: RoleAssistant,
			Content: []Part{
				ToolCallPart{ToolCallID: "call-2", ToolName: "search"},
			},
		},
		ToolMessage(ToolResultPart{ToolCallID: "call-2", ToolName: "search", Output: ToolResultOutput{Type: "text", Value: "kept"}}),
	}

	got := PruneMessages(PruneMessagesOptions{
		Messages:  messages,
		Reasoning: PruneReasoningAll,
		ToolCalls: []PruneToolCallsRule{
			PruneToolCallsBeforeLastMessages(2),
		},
	})

	if len(got) != 4 {
		t.Fatalf("expected empty historical tool message to be removed, got %d messages: %#v", len(got), got)
	}
	if got[1].Role != RoleAssistant {
		t.Fatalf("expected assistant message at index 1, got %#v", got[1])
	}
	if len(got[1].Content) != 1 {
		t.Fatalf("expected reasoning and historical tool call to be pruned, got %#v", got[1].Content)
	}
	if _, ok := got[1].Content[0].(TextPart); !ok {
		t.Fatalf("expected remaining assistant content to be text, got %#v", got[1].Content[0])
	}
	if !containsToolPart(got[2].Content, "call-2") || !containsToolPart(got[3].Content, "call-2") {
		t.Fatalf("expected trailing tool call/result to be retained: %#v", got)
	}
}

func TestPruneMessagesRespectsToolFilter(t *testing.T) {
	messages := []Message{
		{
			Role: RoleAssistant,
			Content: []Part{
				ToolCallPart{ToolCallID: "call-1", ToolName: "weather"},
				ToolCallPart{ToolCallID: "call-2", ToolName: "search"},
			},
		},
		ToolMessage(
			ToolResultPart{ToolCallID: "call-1", ToolName: "weather"},
			ToolResultPart{ToolCallID: "call-2", ToolName: "search"},
		),
	}

	got := PruneMessages(PruneMessagesOptions{
		Messages:  messages,
		ToolCalls: []PruneToolCallsRule{PruneAllToolCalls("weather")},
	})

	if len(got) != 2 {
		t.Fatalf("expected messages containing search parts to remain, got %#v", got)
	}
	if containsToolPart(got[0].Content, "call-1") || containsToolPart(got[1].Content, "call-1") {
		t.Fatalf("expected weather parts to be pruned, got %#v", got)
	}
	if !containsToolPart(got[0].Content, "call-2") || !containsToolPart(got[1].Content, "call-2") {
		t.Fatalf("expected search parts to remain, got %#v", got)
	}
}

func TestResolveToolApproval(t *testing.T) {
	call := ToolCall{ToolCallID: "call-1", ToolName: "weather", Input: map[string]any{"city": "NYC"}}
	tools := map[string]Tool{
		"weather": {
			NeedsApproval: func(_ context.Context, got ToolCall) (ApprovalDecision, error) {
				if !reflect.DeepEqual(got, call) {
					t.Fatalf("unexpected tool call: %#v", got)
				}
				return UserApproval(), nil
			},
		},
	}

	decision, err := ResolveToolApproval(context.Background(), tools, call)
	if err != nil {
		t.Fatalf("ResolveToolApproval failed: %v", err)
	}
	if decision.Type != "user-approval" {
		t.Fatalf("unexpected approval decision: %#v", decision)
	}

	decision, err = ResolveToolApproval(context.Background(), nil, call)
	if err != nil {
		t.Fatalf("ResolveToolApproval without tools failed: %v", err)
	}
	if decision.Type != "not-applicable" {
		t.Fatalf("expected not-applicable, got %#v", decision)
	}
}

func containsToolPart(parts []Part, id string) bool {
	for _, part := range parts {
		if toolPartID(part) == id {
			return true
		}
	}
	return false
}
