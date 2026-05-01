package ai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDecodeChatRequestMatchesHTTPTransportBody(t *testing.T) {
	req, err := DecodeChatRequest(strings.NewReader(`{
		"id": "chat-1",
		"messages": [
			{"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}
		],
		"trigger": "submit-message",
		"messageId": "msg-1",
		"sessionId": "tenant-1"
	}`))
	if err != nil {
		t.Fatalf("DecodeChatRequest failed: %v", err)
	}
	if req.ID != "chat-1" || req.Trigger != "submit-message" || req.MessageID != "msg-1" {
		t.Fatalf("unexpected request metadata: %#v", req)
	}
	if len(req.Messages) != 1 || req.Messages[0].Parts[0].Text != "hi" {
		t.Fatalf("unexpected messages: %#v", req.Messages)
	}
	if req.Body["sessionId"] != "tenant-1" {
		t.Fatalf("unexpected extra body fields: %#v", req.Body)
	}
}

func TestWriteChatUIMessageStreamResponseStreamsModelAsUIMessageSSE(t *testing.T) {
	model := NewMockLanguageModel("chat")
	var captured LanguageModelCallOptions
	model.StreamFunc = func(_ context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		captured = opts
		stream := make(chan StreamPart, 2)
		stream <- StreamPart{Type: "text-delta", TextDelta: "hello"}
		stream <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(stream)
		return &LanguageModelStreamResult{Stream: stream}, nil
	}

	httpReq := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{
		"id": "chat-1",
		"messages": [
			{"id": "msg-1", "role": "user", "parts": [{"type": "text", "text": "hi"}]}
		],
		"trigger": "submit-message",
		"messageId": "msg-1"
	}`))
	rec := httptest.NewRecorder()

	err := WriteChatUIMessageStreamResponse(rec, httpReq, ChatRequestHandlerOptions{
		Stream: StreamTextOptions{GenerateTextOptions: GenerateTextOptions{Model: model}},
	})
	if err != nil {
		t.Fatalf("WriteChatUIMessageStreamResponse failed: %v", err)
	}
	if got := rec.Header().Get("X-Vercel-AI-UI-Message-Stream"); got != "v1" {
		t.Fatalf("unexpected UI stream header: %q", got)
	}
	body := rec.Body.String()
	for _, want := range []string{
		`"type":"start","messageId":"response-2"`,
		`"type":"text-delta","id":"text-1","delta":"hello"`,
		`data: [DONE]`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("response body missing %q:\n%s", want, body)
		}
	}
	if len(captured.Prompt) != 1 || captured.Prompt[0].Role != RoleUser || len(captured.Prompt[0].Content) != 1 {
		t.Fatalf("unexpected model prompt: %#v", captured.Prompt)
	}
	text, ok := captured.Prompt[0].Content[0].(TextPart)
	if !ok || text.Text != "hi" {
		t.Fatalf("unexpected prompt content: %#v", captured.Prompt[0].Content)
	}
}

func TestCreateChatUIMessageStreamResponseReturnsHTTPResponse(t *testing.T) {
	model := NewMockLanguageModel("chat")
	model.StreamFunc = func(context.Context, LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		stream := make(chan StreamPart, 2)
		stream <- StreamPart{Type: "text-delta", TextDelta: "ok"}
		stream <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(stream)
		return &LanguageModelStreamResult{Stream: stream}, nil
	}
	response, err := CreateChatUIMessageStreamResponse(context.Background(), ChatRequest{
		ID: "chat-1",
		Messages: []UIMessage{{
			ID:    "msg-1",
			Role:  RoleUser,
			Parts: []UIPart{{Type: "text", Text: "hi"}},
		}},
	}, ChatRequestHandlerOptions{
		Stream:   StreamTextOptions{GenerateTextOptions: GenerateTextOptions{Model: model}},
		Response: UIMessageStreamResponseOptions{Status: http.StatusAccepted},
	})
	if err != nil {
		t.Fatalf("CreateChatUIMessageStreamResponse failed: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusAccepted {
		t.Fatalf("unexpected response status: %d", response.StatusCode)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("ReadAll failed: %v", err)
	}
	if !strings.Contains(string(body), `"delta":"ok"`) {
		t.Fatalf("unexpected response body: %q", body)
	}
}

func TestWriteChatUIMessageStreamResponseResumesExistingStream(t *testing.T) {
	model := NewMockLanguageModel("chat")
	httpReq := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{
		"id": "chat-1",
		"trigger": "resume-stream",
		"messageId": "msg-1"
	}`))
	rec := httptest.NewRecorder()

	err := WriteChatUIMessageStreamResponse(rec, httpReq, ChatRequestHandlerOptions{
		Stream: StreamTextOptions{GenerateTextOptions: GenerateTextOptions{Model: model}},
		Resume: func(ctx context.Context, req ChatRequest) (<-chan UIMessageChunk, bool, error) {
			if req.ID != "chat-1" || req.MessageID != "msg-1" {
				t.Fatalf("unexpected resume request: %#v", req)
			}
			stream := make(chan UIMessageChunk, 2)
			stream <- StartUIMessageChunk("msg-1")
			stream <- FinishUIMessageChunk(FinishStop)
			close(stream)
			return stream, true, nil
		},
	})
	if err != nil {
		t.Fatalf("WriteChatUIMessageStreamResponse failed: %v", err)
	}
	if len(model.StreamCalls) != 0 {
		t.Fatalf("resume should not start a new model call")
	}
	if !strings.Contains(rec.Body.String(), `"messageId":"msg-1"`) {
		t.Fatalf("unexpected resume response: %s", rec.Body.String())
	}
}

func TestWriteChatUIMessageStreamResponseResumeNoActiveStream(t *testing.T) {
	httpReq := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{
		"id": "chat-1",
		"trigger": "resume-stream"
	}`))
	rec := httptest.NewRecorder()
	err := WriteChatUIMessageStreamResponse(rec, httpReq, ChatRequestHandlerOptions{})
	if err != nil {
		t.Fatalf("WriteChatUIMessageStreamResponse failed: %v", err)
	}
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want 204", rec.Code)
	}
}

func TestWriteCompletionRequestStreamResponseUsesPrompt(t *testing.T) {
	model := NewMockLanguageModel("completion")
	var captured LanguageModelCallOptions
	model.StreamFunc = func(_ context.Context, opts LanguageModelCallOptions) (*LanguageModelStreamResult, error) {
		captured = opts
		stream := make(chan StreamPart, 2)
		stream <- StreamPart{Type: "text-delta", TextDelta: "completed"}
		stream <- StreamPart{Type: "finish", FinishReason: FinishReason{Unified: FinishStop}}
		close(stream)
		return &LanguageModelStreamResult{Stream: stream}, nil
	}

	httpReq := httptest.NewRequest(http.MethodPost, "/api/completion", strings.NewReader(`{"prompt":"finish this","traceId":"abc"}`))
	rec := httptest.NewRecorder()
	err := WriteCompletionRequestStreamResponse(rec, httpReq, CompletionRequestHandlerOptions{
		Stream: StreamTextOptions{GenerateTextOptions: GenerateTextOptions{Model: model}},
	})
	if err != nil {
		t.Fatalf("WriteCompletionRequestStreamResponse failed: %v", err)
	}
	if rec.Body.String() != "completed" {
		t.Fatalf("unexpected completion body: %q", rec.Body.String())
	}
	if len(captured.Prompt) != 1 || captured.Prompt[0].Role != RoleUser {
		t.Fatalf("unexpected completion prompt: %#v", captured.Prompt)
	}
	text, ok := captured.Prompt[0].Content[0].(TextPart)
	if !ok || text.Text != "finish this" {
		t.Fatalf("unexpected completion prompt content: %#v", captured.Prompt[0].Content)
	}
}
