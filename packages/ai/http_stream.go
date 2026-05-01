package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type StreamResponseOptions struct {
	Status     int
	StatusText string
	Headers    map[string]string
}

func WriteTextStreamResponse(w http.ResponseWriter, stream <-chan StreamPart, opts StreamResponseOptions) error {
	prepareStreamHeaders(w, opts, map[string]string{
		"Content-Type":  "text/plain; charset=utf-8",
		"Cache-Control": "no-cache",
	})
	flusher, _ := w.(http.Flusher)
	for part := range stream {
		if part.Err != nil {
			return part.Err
		}
		if part.Type == "text-delta" && part.TextDelta != "" {
			if _, err := w.Write([]byte(part.TextDelta)); err != nil {
				return err
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
	}
	return nil
}

func WriteDataStreamResponse(w http.ResponseWriter, stream <-chan StreamPart, opts StreamResponseOptions) error {
	prepareStreamHeaders(w, opts, map[string]string{
		"Content-Type":  "text/event-stream; charset=utf-8",
		"Cache-Control": "no-cache",
	})
	flusher, _ := w.(http.Flusher)
	for part := range stream {
		if part.Err != nil {
			part.Type = "error"
		}
		data, err := json.Marshal(streamPartPayload(part))
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		if part.Err != nil {
			return part.Err
		}
	}
	return nil
}

func PipeTextStreamToResponse(w http.ResponseWriter, textStream <-chan string, opts StreamResponseOptions) error {
	prepareStreamHeaders(w, opts, map[string]string{
		"Content-Type": "text/plain; charset=utf-8",
	})
	flusher, _ := w.(http.Flusher)
	for text := range textStream {
		if _, err := io.WriteString(w, text); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	return nil
}

func CreateTextStreamResponse(ctx context.Context, textStream <-chan string, opts StreamResponseOptions) *http.Response {
	if ctx == nil {
		ctx = context.Background()
	}
	reader, writer := io.Pipe()
	go func() {
		defer writer.Close()
		for {
			select {
			case <-ctx.Done():
				_ = writer.CloseWithError(ctx.Err())
				return
			case text, ok := <-textStream:
				if !ok {
					return
				}
				if _, err := io.WriteString(writer, text); err != nil {
					_ = writer.CloseWithError(err)
					return
				}
			}
		}
	}()
	return newStreamHTTPResponse(reader, opts, map[string]string{
		"Content-Type": "text/plain; charset=utf-8",
	})
}

func PipeCompletionStreamToResponse(w http.ResponseWriter, completion <-chan string, opts StreamResponseOptions) error {
	return PipeTextStreamToResponse(w, completion, opts)
}

func CreateCompletionStreamResponse(ctx context.Context, completion <-chan string, opts StreamResponseOptions) *http.Response {
	return CreateTextStreamResponse(ctx, completion, opts)
}

func prepareStreamHeaders(w http.ResponseWriter, opts StreamResponseOptions, defaults map[string]string) {
	prepareHeaders(w.Header(), opts.Headers, defaults)
	if opts.Status != 0 {
		w.WriteHeader(opts.Status)
	}
}

func prepareHeaders(header http.Header, headers map[string]string, defaults map[string]string) {
	for key, value := range headers {
		header.Set(key, value)
	}
	for key, value := range defaults {
		if header.Get(key) == "" {
			header.Set(key, value)
		}
	}
}

func newStreamHTTPResponse(body io.ReadCloser, opts StreamResponseOptions, defaults map[string]string) *http.Response {
	status := opts.Status
	if status == 0 {
		status = http.StatusOK
	}
	statusText := opts.StatusText
	if statusText == "" {
		statusText = http.StatusText(status)
	}
	header := http.Header{}
	prepareHeaders(header, opts.Headers, defaults)
	response := &http.Response{
		StatusCode: status,
		Header:     header,
		Body:       body,
	}
	if statusText != "" {
		response.Status = fmt.Sprintf("%d %s", status, statusText)
	} else {
		response.Status = fmt.Sprintf("%d", status)
	}
	return response
}

func streamPartPayload(part StreamPart) map[string]any {
	payload := map[string]any{"type": part.Type}
	if part.ID != "" {
		payload["id"] = part.ID
	}
	if part.TextDelta != "" {
		payload["textDelta"] = part.TextDelta
	}
	if part.ReasoningDelta != "" {
		payload["reasoningDelta"] = part.ReasoningDelta
	}
	if part.ToolCallID != "" {
		payload["toolCallId"] = part.ToolCallID
	}
	if part.ToolName != "" {
		payload["toolName"] = part.ToolName
	}
	if part.ToolInputDelta != "" {
		payload["toolInputDelta"] = part.ToolInputDelta
	}
	if part.ToolInput != "" {
		payload["toolInput"] = part.ToolInput
	}
	if part.FinishReason.Unified != "" || part.FinishReason.Raw != "" {
		payload["finishReason"] = part.FinishReason
	}
	if part.Usage.TotalTokens != nil || part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
		payload["usage"] = part.Usage
	}
	if len(part.Warnings) > 0 {
		payload["warnings"] = part.Warnings
	}
	if part.Request.Body != nil {
		payload["request"] = part.Request
	}
	if part.Response.ID != "" || !part.Response.Timestamp.IsZero() || part.Response.ModelID != "" || len(part.Response.Headers) > 0 || part.Response.Body != nil || len(part.Response.Messages) > 0 {
		payload["response"] = part.Response
	}
	if len(part.ProviderMetadata) > 0 {
		payload["providerMetadata"] = part.ProviderMetadata
	}
	if part.Content != nil {
		payload["contentType"] = part.Content.PartType()
	}
	if part.Raw != nil {
		payload["raw"] = part.Raw
	}
	if part.Err != nil {
		payload["error"] = part.Err.Error()
	}
	return payload
}
