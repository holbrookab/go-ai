package ai

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestProcessTextStreamReadsReaderChunks(t *testing.T) {
	var got string
	err := ProcessTextStream(context.Background(), strings.NewReader("hello"), func(part string) error {
		got += part
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != "hello" {
		t.Fatalf("got %q", got)
	}
}

func TestTransformTextToUIMessageStream(t *testing.T) {
	text := make(chan string, 2)
	text <- "hello "
	text <- "world"
	close(text)

	chunks, err := ReadUIMessageStream(TransformTextToUIMessageStream(context.Background(), text, TextToUIMessageStreamOptions{
		MessageID: "message-1",
		TextID:    "text-1",
	}))
	if err != nil {
		t.Fatal(err)
	}
	gotTypes := []string{}
	for _, chunk := range chunks {
		gotTypes = append(gotTypes, chunk.Type)
	}
	want := []string{
		UIMessageChunkTypeStart,
		UIMessageChunkTypeStartStep,
		UIMessageChunkTypeTextStart,
		UIMessageChunkTypeTextDelta,
		UIMessageChunkTypeTextDelta,
		UIMessageChunkTypeTextEnd,
		UIMessageChunkTypeFinishStep,
		UIMessageChunkTypeFinish,
	}
	if len(gotTypes) != len(want) {
		t.Fatalf("types = %#v, want %#v", gotTypes, want)
	}
	for i := range want {
		if gotTypes[i] != want[i] {
			t.Fatalf("types = %#v, want %#v", gotTypes, want)
		}
	}
}

func TestApplyUIMessageChunkBuildsAssistantMessage(t *testing.T) {
	state := CreateStreamingUIMessageState(nil, "message-1")
	chunks := []UIMessageChunk{
		StartUIMessageChunk("message-1"),
		TextStartUIMessageChunk("text-1"),
		TextDeltaUIMessageChunk("text-1", "hello"),
		TextEndUIMessageChunk("text-1"),
		{Type: UIMessageChunkTypeToolInputAvailable, ToolCallID: "call-1", ToolName: "weather", Input: map[string]any{"city": "NYC"}},
		{Type: UIMessageChunkTypeToolOutputAvailable, ToolCallID: "call-1", Output: "sunny"},
		FinishUIMessageChunk(FinishStop),
	}
	for _, chunk := range chunks {
		if err := ApplyUIMessageChunk(state, chunk); err != nil {
			t.Fatalf("ApplyUIMessageChunk(%s) failed: %v", chunk.Type, err)
		}
	}
	if state.Message.ID != "message-1" || state.Message.Role != RoleAssistant {
		t.Fatalf("unexpected message identity: %#v", state.Message)
	}
	if len(state.Message.Parts) != 2 {
		t.Fatalf("parts = %#v", state.Message.Parts)
	}
	if state.Message.Parts[0].Type != "text" || state.Message.Parts[0].Text != "hello" || state.Message.Parts[0].State != "done" {
		t.Fatalf("unexpected text part: %#v", state.Message.Parts[0])
	}
	if state.Message.Parts[1].Type != "tool-weather" || state.Message.Parts[1].State != "output-available" || state.Message.Parts[1].Output != "sunny" {
		t.Fatalf("unexpected tool part: %#v", state.Message.Parts[1])
	}
	if state.FinishReason != FinishStop {
		t.Fatalf("finish reason = %q", state.FinishReason)
	}
}

func TestProcessUIMessageStreamForwardsChunksAndState(t *testing.T) {
	input := make(chan UIMessageChunk, 4)
	input <- StartUIMessageChunk("message-1")
	input <- TextStartUIMessageChunk("text-1")
	input <- TextDeltaUIMessageChunk("text-1", "hi")
	input <- TextEndUIMessageChunk("text-1")
	close(input)

	out, state := ProcessUIMessageStream(context.Background(), input, ProcessUIMessageStreamOptions{MessageID: "message-1"})
	chunks, err := ReadUIMessageStream(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 4 {
		t.Fatalf("chunks = %#v", chunks)
	}
	if len(state.Message.Parts) != 1 || state.Message.Parts[0].Text != "hi" {
		t.Fatalf("state = %#v", state.Message)
	}
}

func TestProcessUIMessageStreamCallsOnDataForTransientParts(t *testing.T) {
	transient := true
	input := make(chan UIMessageChunk, 1)
	input <- UIMessageChunk{Type: "data-status", ID: "status-1", Data: "working", Transient: &transient}
	close(input)

	var got []UIPart
	out, state := ProcessUIMessageStream(context.Background(), input, ProcessUIMessageStreamOptions{
		OnData: func(part UIPart) {
			got = append(got, part)
		},
	})
	if _, err := ReadUIMessageStream(out); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Type != "data-status" || got[0].Data != "working" {
		t.Fatalf("unexpected data callback parts: %#v", got)
	}
	if len(state.Message.Parts) != 0 {
		t.Fatalf("transient data should not be persisted, got %#v", state.Message.Parts)
	}
}

func TestProcessUIMessageStreamValidatesDataChunkSchema(t *testing.T) {
	input := make(chan UIMessageChunk, 1)
	input <- UIMessageChunk{Type: "data-status", ID: "status-1", Data: map[string]any{"status": 123}}
	close(input)

	var callbackErr error
	out, state := ProcessUIMessageStream(context.Background(), input, ProcessUIMessageStreamOptions{
		DataSchemas: map[string]any{
			"data-status": map[string]any{
				"type":       "object",
				"properties": map[string]any{"status": map[string]any{"type": "string"}},
			},
		},
		OnError: func(err error) {
			callbackErr = err
		},
	})
	chunks, err := ReadUIMessageStream(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0].Type != UIMessageChunkTypeError {
		t.Fatalf("expected validation error chunk, got %#v", chunks)
	}
	if callbackErr == nil || !strings.Contains(callbackErr.Error(), "data validation failed: no object generated: $.status must be string") {
		t.Fatalf("unexpected callback error: %v", callbackErr)
	}
	if len(state.Message.Parts) != 0 {
		t.Fatalf("invalid data should not mutate state, got %#v", state.Message.Parts)
	}
}

func TestProcessUIMessageStreamValidatesMergedMessageMetadata(t *testing.T) {
	input := make(chan UIMessageChunk, 1)
	input <- UIMessageChunk{Type: UIMessageChunkTypeMessageMetadata, MessageMetadata: map[string]any{"tenant": 123}}
	close(input)

	out, _ := ProcessUIMessageStream(context.Background(), input, ProcessUIMessageStreamOptions{
		LastMessage: &UIMessage{
			ID:       "assistant-1",
			Role:     RoleAssistant,
			Metadata: map[string]any{"trace": "abc"},
			Parts:    []UIPart{{Type: "text", Text: "partial"}},
		},
		MessageMetadataSchema: map[string]any{
			"type":       "object",
			"required":   []any{"trace"},
			"properties": map[string]any{"trace": map[string]any{"type": "string"}, "tenant": map[string]any{"type": "string"}},
		},
	})
	chunks, err := ReadUIMessageStream(out, ReadUIMessageStreamOptions{TerminateOnError: true})
	if err == nil {
		t.Fatalf("expected validation error, got chunks %#v", chunks)
	}
	if !errors.Is(err, ErrUIMessageStream) || !strings.Contains(err.Error(), "message metadata validation failed: no object generated: $.tenant must be string") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestHandleUIMessageStreamFinishInjectsIDAndCallsCallbacks(t *testing.T) {
	input := make(chan UIMessageChunk, 6)
	input <- UIMessageChunk{Type: UIMessageChunkTypeStart}
	input <- TextStartUIMessageChunk("text-1")
	input <- TextDeltaUIMessageChunk("text-1", "hello")
	input <- TextEndUIMessageChunk("text-1")
	input <- UIMessageChunk{Type: UIMessageChunkTypeFinishStep}
	input <- FinishUIMessageChunk(FinishStop)
	close(input)

	var steps []UIMessageStreamStepFinishEvent
	var finishes []UIMessageStreamFinishEvent
	out := HandleUIMessageStreamFinish(input, HandleUIMessageStreamFinishOptions{
		MessageID:        "response-id",
		OriginalMessages: []UIMessage{{ID: "user-1", Role: RoleUser, Parts: []UIPart{{Type: "text", Text: "hi"}}}},
		OnStepFinish: func(event UIMessageStreamStepFinishEvent) error {
			event.ResponseMessage.Parts = append(event.ResponseMessage.Parts, UIPart{Type: "text", Text: "mutated"})
			steps = append(steps, event)
			return nil
		},
		OnFinish: func(event UIMessageStreamFinishEvent) error {
			finishes = append(finishes, event)
			return nil
		},
	})
	chunks, err := ReadUIMessageStream(out)
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].MessageID != "response-id" {
		t.Fatalf("expected injected message id, got %#v", chunks[0])
	}
	if len(steps) != 1 || steps[0].IsContinuation {
		t.Fatalf("unexpected step events: %#v", steps)
	}
	if len(finishes) != 1 || finishes[0].ResponseMessage.ID != "response-id" || finishes[0].FinishReason != FinishStop {
		t.Fatalf("unexpected finish events: %#v", finishes)
	}
	if len(finishes[0].ResponseMessage.Parts) != 1 {
		t.Fatalf("finish should not observe step callback mutation: %#v", finishes[0].ResponseMessage)
	}
	if len(finishes[0].Messages) != 2 || finishes[0].Messages[1].ID != "response-id" {
		t.Fatalf("unexpected finish messages: %#v", finishes[0].Messages)
	}
}

func TestHandleUIMessageStreamFinishContinuationAndAbort(t *testing.T) {
	input := make(chan UIMessageChunk, 3)
	input <- UIMessageChunk{Type: UIMessageChunkTypeStart}
	input <- UIMessageChunk{Type: UIMessageChunkTypeAbort, Reason: "manual"}
	input <- FinishUIMessageChunk(FinishStop)
	close(input)

	var finish UIMessageStreamFinishEvent
	out := HandleUIMessageStreamFinish(input, HandleUIMessageStreamFinishOptions{
		MessageID: "ignored",
		OriginalMessages: []UIMessage{
			{ID: "user-1", Role: RoleUser, Parts: []UIPart{{Type: "text", Text: "hi"}}},
			{ID: "assistant-1", Role: RoleAssistant, Parts: []UIPart{{Type: "text", Text: "partial"}}},
		},
		OnFinish: func(event UIMessageStreamFinishEvent) error {
			finish = event
			return nil
		},
	})
	chunks, err := ReadUIMessageStream(out)
	if err != nil {
		t.Fatal(err)
	}
	if chunks[0].MessageID != "assistant-1" {
		t.Fatalf("expected continuation id injection, got %#v", chunks[0])
	}
	if !finish.IsContinuation || !finish.IsAborted || finish.ResponseMessage.ID != "assistant-1" {
		t.Fatalf("unexpected finish event: %#v", finish)
	}
	if len(finish.Messages) != 2 || finish.Messages[1].ID != "assistant-1" {
		t.Fatalf("unexpected continuation messages: %#v", finish.Messages)
	}
}

func TestLastAssistantMessageIsCompleteWithApprovalResponses(t *testing.T) {
	approved := true
	if LastAssistantMessageIsCompleteWithApprovalResponses(nil) {
		t.Fatalf("empty messages should not be complete")
	}
	if LastAssistantMessageIsCompleteWithApprovalResponses([]UIMessage{{ID: "u", Role: RoleUser, Parts: []UIPart{{Type: "text", Text: "hi"}}}}) {
		t.Fatalf("user message should not be complete")
	}
	messages := []UIMessage{{
		ID:   "assistant",
		Role: RoleAssistant,
		Parts: []UIPart{
			{Type: "step-start"},
			{Type: "tool-weather", ToolCallID: "call-1", State: "approval-responded", Approval: &struct {
				ID          string `json:"id"`
				Approved    *bool  `json:"approved,omitempty"`
				Reason      string `json:"reason,omitempty"`
				IsAutomatic bool   `json:"isAutomatic,omitempty"`
			}{ID: "approval-1", Approved: &approved}},
			{Type: "tool-weather", ToolCallID: "call-2", State: "output-available", Output: "sunny"},
		},
	}}
	if !LastAssistantMessageIsCompleteWithApprovalResponses(messages) {
		t.Fatalf("expected approval response completion")
	}
	messages[0].Parts = append(messages[0].Parts, UIPart{Type: "step-start"}, UIPart{Type: "text", Text: "done"})
	if LastAssistantMessageIsCompleteWithApprovalResponses(messages) {
		t.Fatalf("only the last step should be considered")
	}
}
