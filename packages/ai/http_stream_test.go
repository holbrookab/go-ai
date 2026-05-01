package ai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestWriteTextStreamResponse(t *testing.T) {
	stream := make(chan StreamPart, 3)
	stream <- StreamPart{Type: "text-delta", TextDelta: "hello"}
	stream <- StreamPart{Type: "text-delta", TextDelta: " world"}
	close(stream)
	rec := httptest.NewRecorder()
	if err := WriteTextStreamResponse(rec, stream, StreamResponseOptions{}); err != nil {
		t.Fatalf("WriteTextStreamResponse failed: %v", err)
	}
	if rec.Body.String() != "hello world" {
		t.Fatalf("unexpected body: %q", rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Fatalf("unexpected content type: %q", got)
	}
}

func TestWriteDataStreamResponse(t *testing.T) {
	stream := make(chan StreamPart, 2)
	stream <- StreamPart{Type: "text-delta", TextDelta: "hello"}
	close(stream)
	rec := httptest.NewRecorder()
	if err := WriteDataStreamResponse(rec, stream, StreamResponseOptions{Status: http.StatusAccepted}); err != nil {
		t.Fatalf("WriteDataStreamResponse failed: %v", err)
	}
	if rec.Code != http.StatusAccepted {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"textDelta":"hello"`) {
		t.Fatalf("unexpected body: %q", rec.Body.String())
	}
}

func TestWriteDataStreamResponseReturnsStreamError(t *testing.T) {
	stream := make(chan StreamPart, 1)
	stream <- StreamPart{Type: "error", Err: errors.New("boom")}
	close(stream)
	rec := httptest.NewRecorder()
	err := WriteDataStreamResponse(rec, stream, StreamResponseOptions{})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("expected boom error, got %v", err)
	}
}

func TestPipeTextStreamToResponseWritesHeadersAndChunks(t *testing.T) {
	stream := make(chan string, 1)
	stream <- "test-data"
	close(stream)

	rec := httptest.NewRecorder()
	if err := PipeTextStreamToResponse(rec, stream, StreamResponseOptions{
		Status:  http.StatusCreated,
		Headers: map[string]string{"Custom-Header": "test"},
	}); err != nil {
		t.Fatalf("PipeTextStreamToResponse failed: %v", err)
	}
	if rec.Code != http.StatusCreated {
		t.Fatalf("unexpected status: %d", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/plain; charset=utf-8" {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := rec.Header().Get("Custom-Header"); got != "test" {
		t.Fatalf("unexpected custom header: %q", got)
	}
	if rec.Body.String() != "test-data" {
		t.Fatalf("unexpected body: %q", rec.Body.String())
	}
}

func TestCreateTextStreamResponseCreatesReadableResponse(t *testing.T) {
	stream := make(chan string, 2)
	stream <- "hello "
	stream <- "world"
	close(stream)

	response := CreateTextStreamResponse(context.Background(), stream, StreamResponseOptions{
		Status:     http.StatusAccepted,
		StatusText: "Accepted-ish",
		Headers:    map[string]string{"Custom-Header": "test"},
	})
	defer response.Body.Close()

	if response.StatusCode != http.StatusAccepted || response.Status != "202 Accepted-ish" {
		t.Fatalf("unexpected response status: %d %q", response.StatusCode, response.Status)
	}
	if got := response.Header.Get("Content-Type"); got != "text/plain; charset=utf-8" {
		t.Fatalf("unexpected content type: %q", got)
	}
	if got := response.Header.Get("Custom-Header"); got != "test" {
		t.Fatalf("unexpected custom header: %q", got)
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("ReadAll failed: %v", err)
	}
	if string(body) != "hello world" {
		t.Fatalf("unexpected body: %q", body)
	}
}

func TestCompletionStreamResponseAliasesTextStream(t *testing.T) {
	stream := make(chan string, 1)
	stream <- "completion"
	close(stream)

	rec := httptest.NewRecorder()
	if err := PipeCompletionStreamToResponse(rec, stream, StreamResponseOptions{}); err != nil {
		t.Fatalf("PipeCompletionStreamToResponse failed: %v", err)
	}
	if rec.Body.String() != "completion" {
		t.Fatalf("unexpected body: %q", rec.Body.String())
	}
}
