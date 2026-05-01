package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestCreateUIMessageStreamWritesAndCloses(t *testing.T) {
	stream := CreateUIMessageStream(CreateUIMessageStreamOptions{
		Execute: func(writer UIMessageStreamWriter) error {
			writer.Write(TextStartUIMessageChunk("text-1"))
			writer.Write(TextDeltaUIMessageChunk("text-1", "hello"))
			writer.Write(TextEndUIMessageChunk("text-1"))
			return nil
		},
	})

	chunks, err := ReadUIMessageStream(stream)
	if err != nil {
		t.Fatalf("ReadUIMessageStream failed: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %#v", len(chunks), chunks)
	}
	if chunks[1].Type != UIMessageChunkTypeTextDelta || chunks[1].Delta != "hello" {
		t.Fatalf("unexpected text delta chunk: %#v", chunks[1])
	}
}

func TestCreateUIMessageStreamMergesStreams(t *testing.T) {
	merged := make(chan UIMessageChunk, 2)
	stream := CreateUIMessageStream(CreateUIMessageStreamOptions{
		Execute: func(writer UIMessageStreamWriter) error {
			writer.Write(TextDeltaUIMessageChunk("text-1", "before"))
			writer.Merge(merged)
			writer.Write(TextDeltaUIMessageChunk("text-1", "after"))
			merged <- TextDeltaUIMessageChunk("text-2", "merged")
			close(merged)
			return nil
		},
		BufferSize: 3,
	})

	chunks, err := ReadUIMessageStream(stream)
	if err != nil {
		t.Fatalf("ReadUIMessageStream failed: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d: %#v", len(chunks), chunks)
	}
	got := map[string]bool{}
	for _, chunk := range chunks {
		got[chunk.Delta] = true
	}
	for _, delta := range []string{"before", "after", "merged"} {
		if !got[delta] {
			t.Fatalf("missing delta %q from %#v", delta, chunks)
		}
	}
}

func TestCreateUIMessageStreamConvertsExecuteErrorToChunk(t *testing.T) {
	stream := CreateUIMessageStream(CreateUIMessageStreamOptions{
		Execute: func(writer UIMessageStreamWriter) error {
			return errors.New("execute failed")
		},
		OnError: func(error) string {
			return "masked"
		},
	})

	chunks, err := ReadUIMessageStream(stream, ReadUIMessageStreamOptions{TerminateOnError: true})
	if err == nil {
		t.Fatalf("expected error")
	}
	if len(chunks) != 1 || chunks[0].Type != UIMessageChunkTypeError || chunks[0].ErrorText != "masked" {
		t.Fatalf("unexpected error chunks: %#v", chunks)
	}
}

func TestUIMessageChunkHelpers(t *testing.T) {
	if !IsStartUIMessageChunk(StartUIMessageChunk("message-1")) {
		t.Fatalf("expected start chunk")
	}
	if !IsFinishUIMessageChunk(FinishUIMessageChunk(FinishStop)) {
		t.Fatalf("expected finish chunk")
	}
	if !IsErrorUIMessageChunk(ErrorUIMessageChunk(errors.New("boom"))) {
		t.Fatalf("expected error chunk")
	}
	if !IsDataUIMessageChunk(UIMessageChunk{Type: "data-weather"}) {
		t.Fatalf("expected data chunk")
	}
}

func TestGetResponseUIMessageID(t *testing.T) {
	generateID := func() string { return "new-id" }
	if _, ok := GetResponseUIMessageID(GetResponseUIMessageIDOptions{GenerateID: generateID}); ok {
		t.Fatalf("expected no id outside persistence mode")
	}
	got, ok := GetResponseUIMessageID(GetResponseUIMessageIDOptions{
		OriginalMessages:  []UIMessage{{ID: "msg-1", Role: RoleUser, Parts: []UIPart{{Type: "text", Text: "hi"}}}},
		ResponseMessageID: "response-id",
		GenerateID:        generateID,
	})
	if !ok || got != "response-id" {
		t.Fatalf("id = %q, %v", got, ok)
	}
	got, ok = GetResponseUIMessageID(GetResponseUIMessageIDOptions{
		OriginalMessages: []UIMessage{{ID: "assistant-id", Role: RoleAssistant, Parts: []UIPart{{Type: "text", Text: "hi"}}}},
		GenerateID:       generateID,
	})
	if !ok || got != "assistant-id" {
		t.Fatalf("expected assistant continuation id, got %q, %v", got, ok)
	}
	got, ok = GetResponseUIMessageID(GetResponseUIMessageIDOptions{
		OriginalMessages: []UIMessage{},
		GenerateID:       generateID,
	})
	if !ok || got != "new-id" {
		t.Fatalf("expected generated id, got %q, %v", got, ok)
	}
}

func TestValidateUIMessageChunk(t *testing.T) {
	if err := ValidateUIMessageChunk(TextStartUIMessageChunk("text-1")); err != nil {
		t.Fatalf("expected valid chunk, got %v", err)
	}
	if err := ValidateUIMessageChunk(UIMessageChunk{Type: UIMessageChunkTypeToolApprovalResponse, ApprovalID: "approval-1"}); err == nil {
		t.Fatalf("expected missing approved error")
	}
	if err := ValidateUIMessageChunk(UIMessageChunk{Type: "bogus"}); err == nil {
		t.Fatalf("expected unsupported chunk type error")
	}
	if err := ValidateUIMessageChunk(UIMessageChunk{Type: "data-weather", Data: map[string]any{"city": "NYC"}}); err != nil {
		t.Fatalf("expected data chunk to validate, got %v", err)
	}
}

func TestWriteUIMessageStreamResponseSSE(t *testing.T) {
	stream := make(chan UIMessageChunk, 3)
	stream <- TextStartUIMessageChunk("text-1")
	stream <- TextDeltaUIMessageChunk("text-1", "hello")
	stream <- TextEndUIMessageChunk("text-1")
	close(stream)

	rec := httptest.NewRecorder()
	if err := WriteUIMessageStreamResponse(rec, stream, UIMessageStreamResponseOptions{
		Status:  http.StatusAccepted,
		Headers: map[string]string{"Custom-Header": "test"},
	}); err != nil {
		t.Fatalf("WriteUIMessageStreamResponse failed: %v", err)
	}
	if rec.Code != http.StatusAccepted {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := rec.Header().Get("X-Vercel-AI-UI-Message-Stream"); got != "v1" {
		t.Fatalf("unexpected ui stream header: %q", got)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `data: {"type":"text-delta","id":"text-1","delta":"hello"}`) {
		t.Fatalf("unexpected body: %q", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("expected done marker, got %q", body)
	}
}

func TestWriteUIMessageStreamResponseJSONL(t *testing.T) {
	stream := make(chan UIMessageChunk, 1)
	stream <- TextDeltaUIMessageChunk("text-1", "hello")
	close(stream)

	rec := httptest.NewRecorder()
	if err := WriteUIMessageStreamResponse(rec, stream, UIMessageStreamResponseOptions{
		Format: UIMessageStreamFormatJSONL,
	}); err != nil {
		t.Fatalf("WriteUIMessageStreamResponse failed: %v", err)
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("unexpected content type: %q", got)
	}
	if body := rec.Body.String(); body != "{\"type\":\"text-delta\",\"id\":\"text-1\",\"delta\":\"hello\"}\n" {
		t.Fatalf("unexpected body: %q", body)
	}
}

func TestJSONToSSE(t *testing.T) {
	got, err := JSONToSSE(TextDeltaUIMessageChunk("text-1", "hello"))
	if err != nil {
		t.Fatalf("JSONToSSE failed: %v", err)
	}
	want := "data: {\"type\":\"text-delta\",\"id\":\"text-1\",\"delta\":\"hello\"}\n\n"
	if got != want {
		t.Fatalf("event = %q, want %q", got, want)
	}
}

func TestPipeUIMessageStreamToResponseFixtureHeadersAndChunks(t *testing.T) {
	stream := make(chan UIMessageChunk, 3)
	stream <- TextStartUIMessageChunk("1")
	stream <- TextDeltaUIMessageChunk("1", "test-data")
	stream <- TextEndUIMessageChunk("1")
	close(stream)

	consumedDone := make(chan []string, 1)
	rec := httptest.NewRecorder()
	if err := PipeUIMessageStreamToResponse(rec, stream, UIMessageStreamResponseOptions{
		Status:  http.StatusOK,
		Headers: map[string]string{"Custom-Header": "test"},
		ConsumeSSEStream: func(stream <-chan string) {
			var consumed []string
			for event := range stream {
				consumed = append(consumed, event)
			}
			consumedDone <- consumed
		},
	}); err != nil {
		t.Fatalf("PipeUIMessageStreamToResponse failed: %v", err)
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
	headers := rec.Header()
	if got := headers.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := headers.Get("Cache-Control"); got != "no-cache" {
		t.Fatalf("unexpected cache-control: %q", got)
	}
	if got := headers.Get("Connection"); got != "keep-alive" {
		t.Fatalf("unexpected connection: %q", got)
	}
	if got := headers.Get("X-Vercel-AI-UI-Message-Stream"); got != "v1" {
		t.Fatalf("unexpected ui stream header: %q", got)
	}
	if got := headers.Get("X-Accel-Buffering"); got != "no" {
		t.Fatalf("unexpected accel buffering header: %q", got)
	}
	if got := headers.Get("Custom-Header"); got != "test" {
		t.Fatalf("unexpected custom header: %q", got)
	}

	wantChunks := []string{
		"data: {\"type\":\"text-start\",\"id\":\"1\"}\n\n",
		"data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data\"}\n\n",
		"data: {\"type\":\"text-end\",\"id\":\"1\"}\n\n",
		"data: [DONE]\n\n",
	}
	if got := splitSSEBody(rec.Body.String()); !reflect.DeepEqual(got, wantChunks) {
		t.Fatalf("body chunks = %#v, want %#v", got, wantChunks)
	}
	if got := <-consumedDone; !reflect.DeepEqual(got, wantChunks) {
		t.Fatalf("consumed chunks = %#v, want %#v", got, wantChunks)
	}
}

func TestCreateUIMessageStreamResponseCreatesReadableResponse(t *testing.T) {
	stream := make(chan UIMessageChunk, 1)
	stream <- TextDeltaUIMessageChunk("1", "test-data")
	close(stream)

	consumedDone := make(chan []string, 1)
	response := CreateUIMessageStreamResponse(context.Background(), stream, UIMessageStreamResponseOptions{
		Status:     http.StatusAccepted,
		StatusText: "Accepted-ish",
		Headers:    map[string]string{"Custom-Header": "test"},
		ConsumeSSEStream: func(stream <-chan string) {
			var consumed []string
			for event := range stream {
				consumed = append(consumed, event)
			}
			consumedDone <- consumed
		},
	})
	defer response.Body.Close()

	if response.StatusCode != http.StatusAccepted || response.Status != "202 Accepted-ish" {
		t.Fatalf("unexpected response status: %d %q", response.StatusCode, response.Status)
	}
	if got := response.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := response.Header.Get("Custom-Header"); got != "test" {
		t.Fatalf("unexpected custom header: %q", got)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("ReadAll failed: %v", err)
	}
	wantChunks := []string{
		"data: {\"type\":\"text-delta\",\"id\":\"1\",\"delta\":\"test-data\"}\n\n",
		"data: [DONE]\n\n",
	}
	if got := splitSSEBody(string(body)); !reflect.DeepEqual(got, wantChunks) {
		t.Fatalf("body chunks = %#v, want %#v", got, wantChunks)
	}
	if got := <-consumedDone; !reflect.DeepEqual(got, wantChunks) {
		t.Fatalf("consumed chunks = %#v, want %#v", got, wantChunks)
	}
}

func TestPipeUIMessageStreamToResponsePreservesCustomDefaultHeaders(t *testing.T) {
	stream := make(chan UIMessageChunk, 1)
	close(stream)

	rec := httptest.NewRecorder()
	if err := PipeUIMessageStreamToResponse(rec, stream, UIMessageStreamResponseOptions{
		Headers: map[string]string{
			"Content-Type":  "application/custom-stream",
			"Cache-Control": "private",
		},
	}); err != nil {
		t.Fatalf("PipeUIMessageStreamToResponse failed: %v", err)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/custom-stream" {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := rec.Header().Get("Cache-Control"); got != "private" {
		t.Fatalf("unexpected cache-control: %q", got)
	}
}

func splitSSEBody(body string) []string {
	if body == "" {
		return nil
	}
	parts := strings.SplitAfter(body, "\n\n")
	if parts[len(parts)-1] == "" {
		parts = parts[:len(parts)-1]
	}
	return parts
}
