package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

const (
	UIMessageChunkTypeTextStart            = "text-start"
	UIMessageChunkTypeTextDelta            = "text-delta"
	UIMessageChunkTypeTextEnd              = "text-end"
	UIMessageChunkTypeError                = "error"
	UIMessageChunkTypeToolInputStart       = "tool-input-start"
	UIMessageChunkTypeToolInputDelta       = "tool-input-delta"
	UIMessageChunkTypeToolInputAvailable   = "tool-input-available"
	UIMessageChunkTypeToolInputError       = "tool-input-error"
	UIMessageChunkTypeToolApprovalRequest  = "tool-approval-request"
	UIMessageChunkTypeToolApprovalResponse = "tool-approval-response"
	UIMessageChunkTypeToolOutputAvailable  = "tool-output-available"
	UIMessageChunkTypeToolOutputError      = "tool-output-error"
	UIMessageChunkTypeToolOutputDenied     = "tool-output-denied"
	UIMessageChunkTypeReasoningStart       = "reasoning-start"
	UIMessageChunkTypeReasoningDelta       = "reasoning-delta"
	UIMessageChunkTypeReasoningEnd         = "reasoning-end"
	UIMessageChunkTypeCustom               = "custom"
	UIMessageChunkTypeSourceURL            = "source-url"
	UIMessageChunkTypeSourceDocument       = "source-document"
	UIMessageChunkTypeFile                 = "file"
	UIMessageChunkTypeReasoningFile        = "reasoning-file"
	UIMessageChunkTypeStartStep            = "start-step"
	UIMessageChunkTypeFinishStep           = "finish-step"
	UIMessageChunkTypeStart                = "start"
	UIMessageChunkTypeFinish               = "finish"
	UIMessageChunkTypeAbort                = "abort"
	UIMessageChunkTypeMessageMetadata      = "message-metadata"
)

const (
	UIMessageStreamFormatSSE   = "sse"
	UIMessageStreamFormatJSONL = "jsonl"
)

type UIMessageChunk struct {
	Type string `json:"type"`

	ID               string           `json:"id,omitempty"`
	Delta            string           `json:"delta,omitempty"`
	ErrorText        string           `json:"errorText,omitempty"`
	ProviderMetadata ProviderMetadata `json:"providerMetadata,omitempty"`

	ToolCallID       string `json:"toolCallId,omitempty"`
	ToolName         string `json:"toolName,omitempty"`
	InputTextDelta   string `json:"inputTextDelta,omitempty"`
	Input            any    `json:"input,omitempty"`
	Output           any    `json:"output,omitempty"`
	ApprovalID       string `json:"approvalId,omitempty"`
	Approved         *bool  `json:"approved,omitempty"`
	IsAutomatic      *bool  `json:"isAutomatic,omitempty"`
	Reason           string `json:"reason,omitempty"`
	ProviderExecuted *bool  `json:"providerExecuted,omitempty"`
	Dynamic          *bool  `json:"dynamic,omitempty"`
	Preliminary      *bool  `json:"preliminary,omitempty"`
	Title            string `json:"title,omitempty"`

	Kind      string `json:"kind,omitempty"`
	SourceID  string `json:"sourceId,omitempty"`
	URL       string `json:"url,omitempty"`
	MediaType string `json:"mediaType,omitempty"`
	Filename  string `json:"filename,omitempty"`

	Data      any   `json:"data,omitempty"`
	Transient *bool `json:"transient,omitempty"`

	MessageID       string `json:"messageId,omitempty"`
	MessageMetadata any    `json:"messageMetadata,omitempty"`
	FinishReason    string `json:"finishReason,omitempty"`

	Err error `json:"-"`
}

func TextStartUIMessageChunk(id string) UIMessageChunk {
	return UIMessageChunk{Type: UIMessageChunkTypeTextStart, ID: id}
}

func TextDeltaUIMessageChunk(id, delta string) UIMessageChunk {
	return UIMessageChunk{Type: UIMessageChunkTypeTextDelta, ID: id, Delta: delta}
}

func TextEndUIMessageChunk(id string) UIMessageChunk {
	return UIMessageChunk{Type: UIMessageChunkTypeTextEnd, ID: id}
}

func StartUIMessageChunk(messageID string) UIMessageChunk {
	return UIMessageChunk{Type: UIMessageChunkTypeStart, MessageID: messageID}
}

func FinishUIMessageChunk(finishReason string) UIMessageChunk {
	return UIMessageChunk{Type: UIMessageChunkTypeFinish, FinishReason: finishReason}
}

func ErrorUIMessageChunk(err error) UIMessageChunk {
	if err == nil {
		err = errors.New("unknown error")
	}
	return UIMessageChunk{Type: UIMessageChunkTypeError, ErrorText: err.Error(), Err: err}
}

func IsDataUIMessageChunk(chunk UIMessageChunk) bool {
	return strings.HasPrefix(chunk.Type, "data-")
}

func IsStartUIMessageChunk(chunk UIMessageChunk) bool {
	return chunk.Type == UIMessageChunkTypeStart
}

func IsFinishUIMessageChunk(chunk UIMessageChunk) bool {
	return chunk.Type == UIMessageChunkTypeFinish
}

func IsErrorUIMessageChunk(chunk UIMessageChunk) bool {
	return chunk.Type == UIMessageChunkTypeError || chunk.Err != nil
}

type UIMessageStreamWriter struct {
	ctx     context.Context
	out     chan<- UIMessageChunk
	onError func(error) string
	wg      *sync.WaitGroup
}

func (w UIMessageStreamWriter) Write(chunk UIMessageChunk) bool {
	return safeWriteUIMessageChunkContext(w.ctx, w.out, chunk)
}

func (w UIMessageStreamWriter) Merge(stream <-chan UIMessageChunk) bool {
	if stream == nil {
		return false
	}
	w.wg.Add(1)
	go func() {
		defer w.wg.Done()
		for {
			select {
			case <-w.ctx.Done():
				return
			case chunk, ok := <-stream:
				if !ok {
					return
				}
				if !safeWriteUIMessageChunkContext(w.ctx, w.out, chunk) {
					return
				}
			}
		}
	}()
	return true
}

func (w UIMessageStreamWriter) OnError(err error) string {
	if w.onError != nil {
		return w.onError(err)
	}
	return defaultUIMessageStreamErrorText(err)
}

type CreateUIMessageStreamOptions struct {
	Context           context.Context
	Execute           func(writer UIMessageStreamWriter) error
	OnError           func(error) string
	BufferSize        int
	OriginalMessages  []UIMessage
	ResponseMessageID string
	GenerateID        func() string
	OnStepFinish      func(UIMessageStreamStepFinishEvent) error
	OnFinish          func(UIMessageStreamFinishEvent) error
}

func CreateUIMessageStream(opts CreateUIMessageStreamOptions) <-chan UIMessageChunk {
	ctx := opts.Context
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithCancel(ctx)
	bufferSize := opts.BufferSize
	if bufferSize < 0 {
		bufferSize = 0
	}
	out := make(chan UIMessageChunk, bufferSize)
	onError := opts.OnError
	if onError == nil {
		onError = defaultUIMessageStreamErrorText
	}
	var wg sync.WaitGroup
	writer := UIMessageStreamWriter{ctx: ctx, out: out, onError: onError, wg: &wg}

	go func() {
		defer cancel()
		defer close(out)
		if opts.Execute != nil {
			if err := opts.Execute(writer); err != nil {
				safeWriteUIMessageChunkContext(ctx, out, UIMessageChunk{
					Type:      UIMessageChunkTypeError,
					ErrorText: onError(err),
					Err:       err,
				})
			}
		}
		wg.Wait()
	}()

	responseMessageID, hasResponseMessageID := GetResponseUIMessageID(GetResponseUIMessageIDOptions{
		OriginalMessages:  opts.OriginalMessages,
		ResponseMessageID: opts.ResponseMessageID,
		GenerateID:        opts.GenerateID,
	})
	if !hasResponseMessageID && opts.OnStepFinish == nil && opts.OnFinish == nil {
		return out
	}

	return HandleUIMessageStreamFinish(out, HandleUIMessageStreamFinishOptions{
		MessageID:        responseMessageID,
		OriginalMessages: opts.OriginalMessages,
		OnStepFinish:     opts.OnStepFinish,
		OnFinish:         opts.OnFinish,
		OnError: func(err error) {
			if opts.OnError != nil {
				opts.OnError(err)
			}
		},
		Context:    ctx,
		BufferSize: opts.BufferSize,
	})
}

type ReadUIMessageStreamOptions struct {
	TerminateOnError bool
	OnError          func(error)
}

func ReadUIMessageStream(stream <-chan UIMessageChunk, options ...ReadUIMessageStreamOptions) ([]UIMessageChunk, error) {
	var opts ReadUIMessageStreamOptions
	if len(options) > 0 {
		opts = options[0]
	}
	chunks := []UIMessageChunk{}
	for chunk := range stream {
		chunks = append(chunks, chunk)
		if IsErrorUIMessageChunk(chunk) {
			err := chunk.Err
			if err == nil {
				err = NewUIMessageStreamError(chunk.Type, chunk.ID, chunk.ErrorText)
			}
			if opts.OnError != nil {
				opts.OnError(err)
			}
			if opts.TerminateOnError {
				return chunks, err
			}
		}
	}
	return chunks, nil
}

type UIMessageStreamResponseOptions struct {
	Status           int
	StatusText       string
	Headers          map[string]string
	Format           string
	BufferSize       int
	ConsumeSSEStream func(stream <-chan string)
}

type GetResponseUIMessageIDOptions struct {
	OriginalMessages  []UIMessage
	ResponseMessageID string
	GenerateID        func() string
}

func GetResponseUIMessageID(opts GetResponseUIMessageIDOptions) (string, bool) {
	if opts.OriginalMessages == nil {
		return "", false
	}
	if len(opts.OriginalMessages) > 0 {
		lastMessage := opts.OriginalMessages[len(opts.OriginalMessages)-1]
		if lastMessage.Role == RoleAssistant {
			return lastMessage.ID, true
		}
	}
	if opts.ResponseMessageID != "" {
		return opts.ResponseMessageID, true
	}
	if opts.GenerateID != nil {
		return opts.GenerateID(), true
	}
	return fmt.Sprintf("response-%d", len(opts.OriginalMessages)+1), true
}

func ValidateUIMessageChunk(chunk UIMessageChunk) error {
	required := func(name, value string) error {
		if value == "" {
			return NewUIMessageStreamError(chunk.Type, chunkIDForValidation(chunk), fmt.Sprintf("%s is required", name))
		}
		return nil
	}

	switch chunk.Type {
	case UIMessageChunkTypeTextStart, UIMessageChunkTypeTextEnd, UIMessageChunkTypeReasoningStart, UIMessageChunkTypeReasoningEnd:
		return required("id", chunk.ID)
	case UIMessageChunkTypeTextDelta, UIMessageChunkTypeReasoningDelta:
		return required("id", chunk.ID)
	case UIMessageChunkTypeError:
		return nil
	case UIMessageChunkTypeToolInputStart:
		if err := required("toolCallId", chunk.ToolCallID); err != nil {
			return err
		}
		return required("toolName", chunk.ToolName)
	case UIMessageChunkTypeToolInputDelta:
		return required("toolCallId", chunk.ToolCallID)
	case UIMessageChunkTypeToolInputAvailable:
		if err := required("toolCallId", chunk.ToolCallID); err != nil {
			return err
		}
		return required("toolName", chunk.ToolName)
	case UIMessageChunkTypeToolInputError:
		if err := required("toolCallId", chunk.ToolCallID); err != nil {
			return err
		}
		return required("toolName", chunk.ToolName)
	case UIMessageChunkTypeToolApprovalRequest:
		if err := required("approvalId", chunk.ApprovalID); err != nil {
			return err
		}
		return required("toolCallId", chunk.ToolCallID)
	case UIMessageChunkTypeToolApprovalResponse:
		if err := required("approvalId", chunk.ApprovalID); err != nil {
			return err
		}
		if chunk.Approved == nil {
			return NewUIMessageStreamError(chunk.Type, chunk.ApprovalID, "approved is required")
		}
	case UIMessageChunkTypeToolOutputAvailable:
		return required("toolCallId", chunk.ToolCallID)
	case UIMessageChunkTypeToolOutputError:
		return required("toolCallId", chunk.ToolCallID)
	case UIMessageChunkTypeToolOutputDenied:
		return required("toolCallId", chunk.ToolCallID)
	case UIMessageChunkTypeCustom:
		return required("kind", chunk.Kind)
	case UIMessageChunkTypeSourceURL:
		if err := required("sourceId", chunk.SourceID); err != nil {
			return err
		}
		return required("url", chunk.URL)
	case UIMessageChunkTypeSourceDocument:
		if err := required("sourceId", chunk.SourceID); err != nil {
			return err
		}
		if err := required("mediaType", chunk.MediaType); err != nil {
			return err
		}
		return required("title", chunk.Title)
	case UIMessageChunkTypeFile, UIMessageChunkTypeReasoningFile:
		if err := required("url", chunk.URL); err != nil {
			return err
		}
		return required("mediaType", chunk.MediaType)
	case UIMessageChunkTypeStartStep, UIMessageChunkTypeFinishStep, UIMessageChunkTypeStart, UIMessageChunkTypeAbort, UIMessageChunkTypeMessageMetadata:
	case UIMessageChunkTypeFinish:
		switch chunk.FinishReason {
		case "", FinishStop, FinishLength, FinishContentFilter, FinishToolCalls, FinishError, FinishOther:
		default:
			return NewUIMessageStreamError(chunk.Type, chunk.FinishReason, fmt.Sprintf("finishReason %q is not supported", chunk.FinishReason))
		}
	default:
		if !IsDataUIMessageChunk(chunk) {
			return NewUIMessageStreamError(chunk.Type, chunkIDForValidation(chunk), fmt.Sprintf("unsupported chunk type %q", chunk.Type))
		}
	}
	return nil
}

func WriteUIMessageStreamResponse(w http.ResponseWriter, stream <-chan UIMessageChunk, opts UIMessageStreamResponseOptions) error {
	return PipeUIMessageStreamToResponse(w, stream, opts)
}

func JSONToSSE(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("data: %s\n\n", data), nil
}

func WriteJSONToSSE(w io.Writer, value any) error {
	event, err := JSONToSSE(value)
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, event)
	return err
}

func WriteSSEDone(w io.Writer) error {
	_, err := io.WriteString(w, "data: [DONE]\n\n")
	return err
}

func CreateUIMessageStreamResponse(ctx context.Context, stream <-chan UIMessageChunk, opts UIMessageStreamResponseOptions) *http.Response {
	if ctx == nil {
		ctx = context.Background()
	}
	reader, writer := io.Pipe()
	go func() {
		err := writeUIMessageStream(ctx, writer, stream, opts, nil)
		if err != nil {
			_ = writer.CloseWithError(err)
			return
		}
		_ = writer.Close()
	}()
	return newStreamHTTPResponse(reader, streamResponseOptionsFromUI(opts), uiMessageStreamDefaultHeaders(opts.Format))
}

func PipeUIMessageStreamToResponse(w http.ResponseWriter, stream <-chan UIMessageChunk, opts UIMessageStreamResponseOptions) error {
	return PipeUIMessageStreamToResponseContext(context.Background(), w, stream, opts)
}

func PipeUIMessageStreamToResponseContext(ctx context.Context, w http.ResponseWriter, stream <-chan UIMessageChunk, opts UIMessageStreamResponseOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	format := opts.Format
	if format == "" {
		format = UIMessageStreamFormatSSE
	}
	prepareHeaders(w.Header(), opts.Headers, uiMessageStreamDefaultHeaders(format))
	if opts.Status != 0 {
		w.WriteHeader(opts.Status)
	}
	return writeUIMessageStream(ctx, w, stream, opts, w)
}

func writeUIMessageStream(ctx context.Context, w io.Writer, stream <-chan UIMessageChunk, opts UIMessageStreamResponseOptions, responseWriter http.ResponseWriter) error {
	format := opts.Format
	if format == "" {
		format = UIMessageStreamFormatSSE
	}
	flusher, _ := w.(http.Flusher)
	if flusher == nil && responseWriter != nil {
		flusher, _ = responseWriter.(http.Flusher)
	}
	consumer := startSSEConsumer(opts)
	defer consumer.close()

	var streamErr error
	for {
		var chunk UIMessageChunk
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case chunk, ok = <-stream:
			if !ok {
				goto done
			}
		}
		if format == UIMessageStreamFormatJSONL {
			data, err := json.Marshal(chunk)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(w, "%s\n", data); err != nil {
				return err
			}
		} else {
			event, err := JSONToSSE(chunk)
			if err != nil {
				return err
			}
			if !consumer.send(ctx, event) {
				return ctx.Err()
			}
			if _, err := io.WriteString(w, event); err != nil {
				return err
			}
		}
		if flusher != nil {
			flusher.Flush()
		}
		if IsErrorUIMessageChunk(chunk) {
			streamErr = chunk.Err
			if streamErr == nil {
				streamErr = NewUIMessageStreamError(chunk.Type, chunkIDForValidation(chunk), chunk.ErrorText)
			}
			break
		}
	}
done:
	if format == UIMessageStreamFormatSSE {
		done := "data: [DONE]\n\n"
		if !consumer.send(ctx, done) {
			return ctx.Err()
		}
		if _, err := io.WriteString(w, done); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	return streamErr
}

func safeWriteUIMessageChunk(out chan<- UIMessageChunk, chunk UIMessageChunk) {
	defer func() {
		_ = recover()
	}()
	out <- chunk
}

func safeWriteUIMessageChunkContext(ctx context.Context, out chan<- UIMessageChunk, chunk UIMessageChunk) bool {
	defer func() {
		_ = recover()
	}()
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return false
	default:
	}
	select {
	case <-ctx.Done():
		return false
	case out <- chunk:
		return true
	}
}

func defaultUIMessageStreamErrorText(err error) string {
	if err == nil {
		return "unknown error"
	}
	return err.Error()
}

func chunkIDForValidation(chunk UIMessageChunk) string {
	if chunk.ID != "" {
		return chunk.ID
	}
	if chunk.ToolCallID != "" {
		return chunk.ToolCallID
	}
	if chunk.ApprovalID != "" {
		return chunk.ApprovalID
	}
	if chunk.SourceID != "" {
		return chunk.SourceID
	}
	return chunk.Type
}

type sseConsumer struct {
	ch chan string
}

func startSSEConsumer(opts UIMessageStreamResponseOptions) *sseConsumer {
	if opts.ConsumeSSEStream == nil {
		return nil
	}
	bufferSize := opts.BufferSize
	if bufferSize == 0 {
		bufferSize = 16
	}
	if bufferSize < 0 {
		bufferSize = 0
	}
	ch := make(chan string, bufferSize)
	go func() {
		defer func() {
			_ = recover()
		}()
		opts.ConsumeSSEStream(ch)
	}()
	return &sseConsumer{ch: ch}
}

func (c *sseConsumer) send(ctx context.Context, event string) bool {
	if c == nil {
		return true
	}
	defer func() {
		_ = recover()
	}()
	select {
	case <-ctx.Done():
		return false
	case c.ch <- event:
		return true
	}
}

func (c *sseConsumer) close() {
	if c == nil {
		return
	}
	close(c.ch)
}

func uiMessageStreamDefaultHeaders(format string) map[string]string {
	if format == UIMessageStreamFormatJSONL {
		return map[string]string{
			"Content-Type":  "application/x-ndjson; charset=utf-8",
			"Cache-Control": "no-cache",
		}
	}
	return map[string]string{
		"Content-Type":                  "text/event-stream",
		"Cache-Control":                 "no-cache",
		"Connection":                    "keep-alive",
		"X-Vercel-AI-UI-Message-Stream": "v1",
		"X-Accel-Buffering":             "no",
	}
}

func streamResponseOptionsFromUI(opts UIMessageStreamResponseOptions) StreamResponseOptions {
	return StreamResponseOptions{
		Status:     opts.Status,
		StatusText: opts.StatusText,
		Headers:    opts.Headers,
	}
}
