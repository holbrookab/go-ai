package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

type ChatRequest struct {
	ID        string
	Messages  []UIMessage
	Trigger   string
	MessageID string
	Body      map[string]any
}

type CompletionRequest struct {
	Prompt string
	Body   map[string]any
}

type ChatRequestHandlerOptions struct {
	Stream    StreamTextOptions
	Validate  ValidateUIMessagesOptions
	Convert   ConvertToModelMessagesOptions
	Response  UIMessageStreamResponseOptions
	MessageID string
	TextID    string
}

type CompletionRequestHandlerOptions struct {
	Stream   StreamTextOptions
	Response StreamResponseOptions
}

func DecodeChatRequest(body io.Reader) (ChatRequest, error) {
	raw, err := decodeRequestObject(body)
	if err != nil {
		return ChatRequest{}, err
	}
	var req ChatRequest
	if err := decodeOptionalField(raw, "id", &req.ID); err != nil {
		return ChatRequest{}, err
	}
	if err := decodeRequiredField(raw, "messages", &req.Messages); err != nil {
		return ChatRequest{}, err
	}
	if err := decodeOptionalField(raw, "trigger", &req.Trigger); err != nil {
		return ChatRequest{}, err
	}
	if err := decodeOptionalField(raw, "messageId", &req.MessageID); err != nil {
		return ChatRequest{}, err
	}
	req.Body, err = decodeExtraFields(raw, "id", "messages", "trigger", "messageId")
	if err != nil {
		return ChatRequest{}, err
	}
	return req, nil
}

func DecodeCompletionRequest(body io.Reader) (CompletionRequest, error) {
	raw, err := decodeRequestObject(body)
	if err != nil {
		return CompletionRequest{}, err
	}
	var req CompletionRequest
	if err := decodeRequiredField(raw, "prompt", &req.Prompt); err != nil {
		return CompletionRequest{}, err
	}
	req.Body, err = decodeExtraFields(raw, "prompt")
	if err != nil {
		return CompletionRequest{}, err
	}
	return req, nil
}

func CreateChatUIMessageStream(ctx context.Context, req ChatRequest, opts ChatRequestHandlerOptions) (<-chan UIMessageChunk, error) {
	validateOpts := opts.Validate
	if validateOpts.Tools == nil {
		validateOpts.Tools = opts.Stream.Tools
	}
	if err := ValidateUIMessages(req.Messages, validateOpts); err != nil {
		return nil, err
	}

	convertOpts := opts.Convert
	if convertOpts.Tools == nil {
		convertOpts.Tools = opts.Stream.Tools
	}
	modelMessages, err := ConvertToModelMessages(req.Messages, convertOpts)
	if err != nil {
		return nil, err
	}

	streamOpts := opts.Stream
	streamOpts.Messages = modelMessages
	streamOpts.Prompt = ""
	result, err := StreamText(ctx, streamOpts)
	if err != nil {
		return nil, err
	}

	messageID := opts.MessageID
	if messageID == "" {
		messageID = responseMessageIDFromChatRequest(req)
	}
	return CreateStreamTextUIMessageStream(ctx, result, StreamTextUIMessageStreamOptions{
		MessageID:  messageID,
		TextID:     opts.TextID,
		BufferSize: opts.Response.BufferSize,
	}), nil
}

func CreateChatUIMessageStreamResponse(ctx context.Context, req ChatRequest, opts ChatRequestHandlerOptions) (*http.Response, error) {
	stream, err := CreateChatUIMessageStream(ctx, req, opts)
	if err != nil {
		return nil, err
	}
	return CreateUIMessageStreamResponse(ctx, stream, opts.Response), nil
}

func WriteChatUIMessageStreamResponse(w http.ResponseWriter, r *http.Request, opts ChatRequestHandlerOptions) error {
	if err := requirePost(r); err != nil {
		return err
	}
	req, err := DecodeChatRequest(r.Body)
	if err != nil {
		return err
	}
	stream, err := CreateChatUIMessageStream(r.Context(), req, opts)
	if err != nil {
		return err
	}
	return WriteUIMessageStreamResponse(w, stream, opts.Response)
}

func CreateCompletionStream(ctx context.Context, req CompletionRequest, opts CompletionRequestHandlerOptions) (*StreamTextResult, error) {
	streamOpts := opts.Stream
	streamOpts.Prompt = req.Prompt
	streamOpts.Messages = nil
	return StreamText(ctx, streamOpts)
}

func WriteCompletionRequestStreamResponse(w http.ResponseWriter, r *http.Request, opts CompletionRequestHandlerOptions) error {
	if err := requirePost(r); err != nil {
		return err
	}
	req, err := DecodeCompletionRequest(r.Body)
	if err != nil {
		return err
	}
	result, err := CreateCompletionStream(r.Context(), req, opts)
	if err != nil {
		return err
	}
	return WriteTextStreamResponse(w, result.Stream, opts.Response)
}

func decodeRequestObject(body io.Reader) (map[string]json.RawMessage, error) {
	if body == nil {
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: "request body is required"}
	}
	decoder := json.NewDecoder(body)
	raw := map[string]json.RawMessage{}
	if err := decoder.Decode(&raw); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: "request body is required"}
		}
		return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("invalid JSON request body: %v", err)}
	}
	return raw, nil
}

func decodeRequiredField(raw map[string]json.RawMessage, name string, out any) error {
	value, ok := raw[name]
	if !ok {
		return &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("%s is required", name)}
	}
	if err := json.Unmarshal(value, out); err != nil {
		return &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("invalid %s: %v", name, err)}
	}
	return nil
}

func decodeOptionalField(raw map[string]json.RawMessage, name string, out any) error {
	value, ok := raw[name]
	if !ok {
		return nil
	}
	if err := json.Unmarshal(value, out); err != nil {
		return &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("invalid %s: %v", name, err)}
	}
	return nil
}

func decodeExtraFields(raw map[string]json.RawMessage, known ...string) (map[string]any, error) {
	knownSet := map[string]bool{}
	for _, key := range known {
		knownSet[key] = true
	}
	extra := map[string]any{}
	for key, value := range raw {
		if knownSet[key] {
			continue
		}
		var decoded any
		if err := json.Unmarshal(value, &decoded); err != nil {
			return nil, &SDKError{Kind: ErrInvalidArgument, Message: fmt.Sprintf("invalid %s: %v", key, err)}
		}
		extra[key] = decoded
	}
	if len(extra) == 0 {
		return nil, nil
	}
	return extra, nil
}

func requirePost(r *http.Request) error {
	if r == nil {
		return &SDKError{Kind: ErrInvalidArgument, Message: "request is required"}
	}
	if r.Method != "" && r.Method != http.MethodPost {
		return &SDKError{Kind: ErrInvalidArgument, Message: "request method must be POST"}
	}
	return nil
}

func responseMessageIDFromChatRequest(req ChatRequest) string {
	messageID, _ := GetResponseUIMessageID(GetResponseUIMessageIDOptions{OriginalMessages: req.Messages})
	return messageID
}
