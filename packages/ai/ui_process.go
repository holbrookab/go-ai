package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const defaultUITextPartID = "text-1"

type TextToUIMessageStreamOptions struct {
	MessageID  string
	TextID     string
	BufferSize int
}

func ProcessTextStream(ctx context.Context, reader io.Reader, onTextPart func(string) error) error {
	if reader == nil {
		return nil
	}
	if onTextPart == nil {
		onTextPart = func(string) error { return nil }
	}

	buf := make([]byte, 32*1024)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, err := reader.Read(buf)
		if n > 0 {
			if callbackErr := onTextPart(string(buf[:n])); callbackErr != nil {
				return callbackErr
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func CollectTextStream(ctx context.Context, stream <-chan string) (string, error) {
	var text strings.Builder
	for {
		select {
		case <-ctx.Done():
			return text.String(), ctx.Err()
		case part, ok := <-stream:
			if !ok {
				return text.String(), nil
			}
			text.WriteString(part)
		}
	}
}

func TransformTextToUIMessageStream(ctx context.Context, stream <-chan string, options ...TextToUIMessageStreamOptions) <-chan UIMessageChunk {
	opts := firstTextToUIMessageStreamOptions(options)
	return CreateUIMessageStream(CreateUIMessageStreamOptions{
		BufferSize: opts.BufferSize,
		Execute: func(writer UIMessageStreamWriter) error {
			writeTextUIMessageStreamStart(writer, opts)
			for {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case part, ok := <-stream:
					if !ok {
						writeTextUIMessageStreamFinish(writer, opts)
						return nil
					}
					writer.Write(TextDeltaUIMessageChunk(textToUIMessageTextID(opts), part))
				}
			}
		},
	})
}

func TextReaderToUIMessageStream(ctx context.Context, reader io.Reader, options ...TextToUIMessageStreamOptions) <-chan UIMessageChunk {
	opts := firstTextToUIMessageStreamOptions(options)
	return CreateUIMessageStream(CreateUIMessageStreamOptions{
		BufferSize: opts.BufferSize,
		Execute: func(writer UIMessageStreamWriter) error {
			writeTextUIMessageStreamStart(writer, opts)
			if err := ProcessTextStream(ctx, reader, func(part string) error {
				writer.Write(TextDeltaUIMessageChunk(textToUIMessageTextID(opts), part))
				return nil
			}); err != nil {
				return err
			}
			writeTextUIMessageStreamFinish(writer, opts)
			return nil
		},
	})
}

type StreamingUIMessageState struct {
	Message              UIMessage
	ActiveTextParts      map[string]int
	ActiveReasoningParts map[string]int
	PartialToolCalls     map[string]PartialToolCallState
	FinishReason         string
}

type PartialToolCallState struct {
	Text     string
	ToolName string
	Dynamic  bool
	Title    string
}

func CreateStreamingUIMessageState(lastMessage *UIMessage, messageID string) *StreamingUIMessageState {
	var message UIMessage
	if lastMessage != nil && lastMessage.Role == RoleAssistant {
		message = *lastMessage
		message.Parts = append([]UIPart(nil), lastMessage.Parts...)
	} else {
		message = UIMessage{ID: messageID, Role: RoleAssistant, Parts: []UIPart{}}
	}
	return &StreamingUIMessageState{
		Message:              message,
		ActiveTextParts:      map[string]int{},
		ActiveReasoningParts: map[string]int{},
		PartialToolCalls:     map[string]PartialToolCallState{},
	}
}

type ProcessUIMessageStreamOptions struct {
	LastMessage           *UIMessage
	MessageID             string
	MessageMetadataSchema any
	DataSchemas           map[string]any
	OnError               func(error)
	OnToolCall            func(UIMessageChunk) error
	OnData                func(UIPart)
	BufferSize            int
}

type UIMessageStreamStepFinishEvent struct {
	IsContinuation  bool
	ResponseMessage UIMessage
	Messages        []UIMessage
}

type UIMessageStreamFinishEvent struct {
	IsAborted       bool
	IsContinuation  bool
	ResponseMessage UIMessage
	Messages        []UIMessage
	FinishReason    string
}

type HandleUIMessageStreamFinishOptions struct {
	MessageID        string
	OriginalMessages []UIMessage
	OnError          func(error)
	OnStepFinish     func(UIMessageStreamStepFinishEvent) error
	OnFinish         func(UIMessageStreamFinishEvent) error
	BufferSize       int
}

func ProcessUIMessageStream(ctx context.Context, stream <-chan UIMessageChunk, opts ProcessUIMessageStreamOptions) (<-chan UIMessageChunk, *StreamingUIMessageState) {
	state := CreateStreamingUIMessageState(opts.LastMessage, opts.MessageID)
	out := make(chan UIMessageChunk, opts.BufferSize)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				safeWriteUIMessageChunk(out, ErrorUIMessageChunk(ctx.Err()))
				if opts.OnError != nil {
					opts.OnError(ctx.Err())
				}
				return
			case chunk, ok := <-stream:
				if !ok {
					return
				}
				if err := validateUIMessageStreamChunkSchemas(state, chunk, opts); err != nil {
					safeWriteUIMessageChunk(out, ErrorUIMessageChunk(err))
					if opts.OnError != nil {
						opts.OnError(err)
					}
					continue
				}
				if err := ApplyUIMessageChunk(state, chunk); err != nil {
					safeWriteUIMessageChunk(out, ErrorUIMessageChunk(err))
					if opts.OnError != nil {
						opts.OnError(err)
					}
					continue
				}
				if chunk.Type == UIMessageChunkTypeError {
					err := chunk.Err
					if err == nil {
						err = NewUIMessageStreamError(chunk.Type, chunk.ID, chunk.ErrorText)
					}
					if opts.OnError != nil {
						opts.OnError(err)
					}
				}
				if chunk.Type == UIMessageChunkTypeToolInputAvailable && opts.OnToolCall != nil && !boolPtrValue(chunk.ProviderExecuted) {
					if err := opts.OnToolCall(chunk); err != nil {
						safeWriteUIMessageChunk(out, ErrorUIMessageChunk(err))
						if opts.OnError != nil {
							opts.OnError(err)
						}
						continue
					}
				}
				if IsDataUIMessageChunk(chunk) && opts.OnData != nil {
					opts.OnData(UIPart{Type: chunk.Type, ID: chunk.ID, Data: chunk.Data})
				}
				safeWriteUIMessageChunk(out, chunk)
			}
		}
	}()
	return out, state
}

func HandleUIMessageStreamFinish(stream <-chan UIMessageChunk, opts HandleUIMessageStreamFinishOptions) <-chan UIMessageChunk {
	out := make(chan UIMessageChunk, opts.BufferSize)
	go func() {
		defer close(out)

		originalMessages := cloneUIMessages(opts.OriginalMessages)
		lastMessage := lastAssistantMessage(originalMessages)
		messageID := opts.MessageID
		if lastMessage != nil {
			messageID = lastMessage.ID
		}

		processCallbacks := opts.OnStepFinish != nil || opts.OnFinish != nil
		state := CreateStreamingUIMessageState(lastMessage, messageID)
		isAborted := false
		finishCalled := false

		callOnStepFinish := func() {
			if opts.OnStepFinish == nil {
				return
			}
			if err := opts.OnStepFinish(UIMessageStreamStepFinishEvent{
				IsContinuation:  isContinuation(originalMessages, state.Message),
				ResponseMessage: cloneUIMessage(state.Message),
				Messages:        messagesWithResponse(originalMessages, state.Message),
			}); err != nil && opts.OnError != nil {
				opts.OnError(err)
			}
		}

		callOnFinish := func() {
			if finishCalled || opts.OnFinish == nil {
				return
			}
			finishCalled = true
			if err := opts.OnFinish(UIMessageStreamFinishEvent{
				IsAborted:       isAborted,
				IsContinuation:  isContinuation(originalMessages, state.Message),
				ResponseMessage: cloneUIMessage(state.Message),
				Messages:        messagesWithResponse(originalMessages, state.Message),
				FinishReason:    state.FinishReason,
			}); err != nil && opts.OnError != nil {
				opts.OnError(err)
			}
		}
		defer callOnFinish()

		for chunk := range stream {
			if chunk.Type == UIMessageChunkTypeStart && chunk.MessageID == "" && messageID != "" {
				chunk.MessageID = messageID
			}
			if chunk.Type == UIMessageChunkTypeAbort {
				isAborted = true
			}
			if processCallbacks {
				if err := ValidateUIMessageChunk(chunk); err != nil {
					safeWriteUIMessageChunk(out, ErrorUIMessageChunk(err))
					if opts.OnError != nil {
						opts.OnError(err)
					}
					continue
				}
				if err := ApplyUIMessageChunk(state, chunk); err != nil {
					safeWriteUIMessageChunk(out, ErrorUIMessageChunk(err))
					if opts.OnError != nil {
						opts.OnError(err)
					}
					continue
				}
				if chunk.Type == UIMessageChunkTypeError {
					err := chunk.Err
					if err == nil {
						err = errors.New(chunk.ErrorText)
					}
					if opts.OnError != nil {
						opts.OnError(err)
					}
				}
				if chunk.Type == UIMessageChunkTypeFinishStep {
					callOnStepFinish()
				}
			}
			safeWriteUIMessageChunk(out, chunk)
		}
	}()
	return out
}

func ApplyUIMessageChunk(state *StreamingUIMessageState, chunk UIMessageChunk) error {
	if state == nil {
		return NewUIMessageStreamError(chunk.Type, chunk.ID, "streaming UI message state is nil")
	}
	ensureStreamingUIMessageState(state)

	switch chunk.Type {
	case UIMessageChunkTypeTextStart:
		part := UIPart{Type: "text", State: "streaming", ProviderMetadata: chunk.ProviderMetadata}
		state.Message.Parts = append(state.Message.Parts, part)
		state.ActiveTextParts[chunk.ID] = len(state.Message.Parts) - 1
	case UIMessageChunkTypeTextDelta:
		index, ok := state.ActiveTextParts[chunk.ID]
		if !ok || index >= len(state.Message.Parts) {
			return NewUIMessageStreamError(chunk.Type, chunk.ID, fmt.Sprintf("Received text-delta for missing text part with ID %q. Ensure a text-start chunk is sent first.", chunk.ID))
		}
		state.Message.Parts[index].Text += chunk.Delta
		state.Message.Parts[index].ProviderMetadata = mergeMetadata(state.Message.Parts[index].ProviderMetadata, chunk.ProviderMetadata)
	case UIMessageChunkTypeTextEnd:
		index, ok := state.ActiveTextParts[chunk.ID]
		if !ok || index >= len(state.Message.Parts) {
			return NewUIMessageStreamError(chunk.Type, chunk.ID, fmt.Sprintf("Received text-end for missing text part with ID %q. Ensure a text-start chunk is sent first.", chunk.ID))
		}
		state.Message.Parts[index].State = "done"
		state.Message.Parts[index].ProviderMetadata = mergeMetadata(state.Message.Parts[index].ProviderMetadata, chunk.ProviderMetadata)
		delete(state.ActiveTextParts, chunk.ID)
	case UIMessageChunkTypeReasoningStart:
		part := UIPart{Type: "reasoning", State: "streaming", ProviderMetadata: chunk.ProviderMetadata}
		state.Message.Parts = append(state.Message.Parts, part)
		state.ActiveReasoningParts[chunk.ID] = len(state.Message.Parts) - 1
	case UIMessageChunkTypeReasoningDelta:
		index, ok := state.ActiveReasoningParts[chunk.ID]
		if !ok || index >= len(state.Message.Parts) {
			return NewUIMessageStreamError(chunk.Type, chunk.ID, fmt.Sprintf("Received reasoning-delta for missing reasoning part with ID %q. Ensure a reasoning-start chunk is sent first.", chunk.ID))
		}
		state.Message.Parts[index].Text += chunk.Delta
		state.Message.Parts[index].ProviderMetadata = mergeMetadata(state.Message.Parts[index].ProviderMetadata, chunk.ProviderMetadata)
	case UIMessageChunkTypeReasoningEnd:
		index, ok := state.ActiveReasoningParts[chunk.ID]
		if !ok || index >= len(state.Message.Parts) {
			return NewUIMessageStreamError(chunk.Type, chunk.ID, fmt.Sprintf("Received reasoning-end for missing reasoning part with ID %q. Ensure a reasoning-start chunk is sent first.", chunk.ID))
		}
		state.Message.Parts[index].State = "done"
		state.Message.Parts[index].ProviderMetadata = mergeMetadata(state.Message.Parts[index].ProviderMetadata, chunk.ProviderMetadata)
		delete(state.ActiveReasoningParts, chunk.ID)
	case UIMessageChunkTypeCustom:
		state.Message.Parts = append(state.Message.Parts, UIPart{Type: "custom", Kind: chunk.Kind, ProviderMetadata: chunk.ProviderMetadata})
	case UIMessageChunkTypeFile, UIMessageChunkTypeReasoningFile:
		state.Message.Parts = append(state.Message.Parts, UIPart{Type: chunk.Type, MediaType: chunk.MediaType, Filename: chunk.Filename, URL: chunk.URL, ProviderMetadata: chunk.ProviderMetadata})
	case UIMessageChunkTypeSourceURL:
		state.Message.Parts = append(state.Message.Parts, UIPart{Type: chunk.Type, SourceID: chunk.SourceID, URL: chunk.URL, Title: chunk.Title, ProviderMetadata: chunk.ProviderMetadata})
	case UIMessageChunkTypeSourceDocument:
		state.Message.Parts = append(state.Message.Parts, UIPart{Type: chunk.Type, SourceID: chunk.SourceID, MediaType: chunk.MediaType, Filename: chunk.Filename, Title: chunk.Title, ProviderMetadata: chunk.ProviderMetadata})
	case UIMessageChunkTypeToolInputStart:
		dynamic := boolPtrValue(chunk.Dynamic)
		state.PartialToolCalls[chunk.ToolCallID] = PartialToolCallState{ToolName: chunk.ToolName, Dynamic: dynamic, Title: chunk.Title}
		updateStreamingToolPart(state, streamingToolPartUpdate{
			ToolCallID:       chunk.ToolCallID,
			ToolName:         chunk.ToolName,
			Dynamic:          dynamic,
			State:            "input-streaming",
			Input:            nil,
			ProviderExecuted: chunk.ProviderExecuted,
			ProviderMetadata: chunk.ProviderMetadata,
			Title:            chunk.Title,
		})
	case UIMessageChunkTypeToolInputDelta:
		partial, ok := state.PartialToolCalls[chunk.ToolCallID]
		if !ok {
			return NewUIMessageStreamError(chunk.Type, chunk.ToolCallID, fmt.Sprintf("Received tool-input-delta for missing tool call with ID %q. Ensure a tool-input-start chunk is sent first.", chunk.ToolCallID))
		}
		partial.Text += chunk.InputTextDelta
		state.PartialToolCalls[chunk.ToolCallID] = partial
		updateStreamingToolPart(state, streamingToolPartUpdate{
			ToolCallID: chunk.ToolCallID,
			ToolName:   partial.ToolName,
			Dynamic:    partial.Dynamic,
			State:      "input-streaming",
			Input:      parsePragmaticPartialJSON(partial.Text),
			Title:      partial.Title,
		})
	case UIMessageChunkTypeToolInputAvailable:
		updateStreamingToolPart(state, streamingToolPartUpdate{
			ToolCallID:       chunk.ToolCallID,
			ToolName:         chunk.ToolName,
			Dynamic:          boolPtrValue(chunk.Dynamic),
			State:            "input-available",
			Input:            chunk.Input,
			ProviderExecuted: chunk.ProviderExecuted,
			ProviderMetadata: chunk.ProviderMetadata,
			Title:            chunk.Title,
		})
		delete(state.PartialToolCalls, chunk.ToolCallID)
	case UIMessageChunkTypeToolInputError:
		existing, found := findToolPartIndex(state.Message.Parts, chunk.ToolCallID)
		dynamic := boolPtrValue(chunk.Dynamic)
		if found {
			dynamic = IsDynamicToolUIPart(state.Message.Parts[existing])
		}
		update := streamingToolPartUpdate{
			ToolCallID:       chunk.ToolCallID,
			ToolName:         chunk.ToolName,
			Dynamic:          dynamic,
			State:            "output-error",
			Input:            chunk.Input,
			ErrorText:        chunk.ErrorText,
			ProviderExecuted: chunk.ProviderExecuted,
			ProviderMetadata: chunk.ProviderMetadata,
			Title:            chunk.Title,
		}
		if !dynamic {
			update.Input = nil
			update.RawInput = chunk.Input
		}
		updateStreamingToolPart(state, update)
	case UIMessageChunkTypeToolApprovalRequest:
		part, err := getStreamingToolPart(state, chunk.ToolCallID)
		if err != nil {
			return err
		}
		part.State = "approval-requested"
		part.Approval = &struct {
			ID          string `json:"id"`
			Approved    *bool  `json:"approved,omitempty"`
			Reason      string `json:"reason,omitempty"`
			IsAutomatic bool   `json:"isAutomatic,omitempty"`
		}{ID: chunk.ApprovalID, IsAutomatic: boolPtrValue(chunk.IsAutomatic)}
	case UIMessageChunkTypeToolApprovalResponse:
		part, err := getStreamingToolPartByApprovalID(state, chunk.ApprovalID)
		if err != nil {
			return err
		}
		automatic := part.Approval != nil && part.Approval.IsAutomatic
		part.State = "approval-responded"
		part.Approval = &struct {
			ID          string `json:"id"`
			Approved    *bool  `json:"approved,omitempty"`
			Reason      string `json:"reason,omitempty"`
			IsAutomatic bool   `json:"isAutomatic,omitempty"`
		}{ID: chunk.ApprovalID, Approved: chunk.Approved, Reason: chunk.Reason, IsAutomatic: automatic}
		if chunk.ProviderExecuted != nil {
			part.ProviderExecuted = *chunk.ProviderExecuted
		}
		part.CallProviderMetadata = mergeMetadata(part.CallProviderMetadata, chunk.ProviderMetadata)
	case UIMessageChunkTypeToolOutputDenied:
		part, err := getStreamingToolPart(state, chunk.ToolCallID)
		if err != nil {
			return err
		}
		part.State = "output-denied"
	case UIMessageChunkTypeToolOutputAvailable:
		part, err := getStreamingToolPart(state, chunk.ToolCallID)
		if err != nil {
			return err
		}
		updateStreamingToolPart(state, streamingToolPartUpdate{
			ToolCallID:       chunk.ToolCallID,
			ToolName:         ToolName(*part),
			Dynamic:          IsDynamicToolUIPart(*part),
			State:            "output-available",
			Input:            part.Input,
			Output:           chunk.Output,
			Preliminary:      chunk.Preliminary,
			ProviderExecuted: chunk.ProviderExecuted,
			ProviderMetadata: chunk.ProviderMetadata,
			Title:            part.Title,
		})
	case UIMessageChunkTypeToolOutputError:
		part, err := getStreamingToolPart(state, chunk.ToolCallID)
		if err != nil {
			return err
		}
		updateStreamingToolPart(state, streamingToolPartUpdate{
			ToolCallID:       chunk.ToolCallID,
			ToolName:         ToolName(*part),
			Dynamic:          IsDynamicToolUIPart(*part),
			State:            "output-error",
			Input:            part.Input,
			RawInput:         part.RawInput,
			ErrorText:        chunk.ErrorText,
			ProviderExecuted: chunk.ProviderExecuted,
			ProviderMetadata: chunk.ProviderMetadata,
			Title:            part.Title,
		})
	case UIMessageChunkTypeStartStep:
		state.Message.Parts = append(state.Message.Parts, UIPart{Type: "step-start"})
	case UIMessageChunkTypeFinishStep:
		state.ActiveTextParts = map[string]int{}
		state.ActiveReasoningParts = map[string]int{}
	case UIMessageChunkTypeStart:
		if chunk.MessageID != "" {
			state.Message.ID = chunk.MessageID
		}
		state.Message.Metadata = mergeUIMessageMetadata(state.Message.Metadata, chunk.MessageMetadata)
	case UIMessageChunkTypeFinish:
		if chunk.FinishReason != "" {
			state.FinishReason = chunk.FinishReason
		}
		state.Message.Metadata = mergeUIMessageMetadata(state.Message.Metadata, chunk.MessageMetadata)
	case UIMessageChunkTypeMessageMetadata:
		state.Message.Metadata = mergeUIMessageMetadata(state.Message.Metadata, chunk.MessageMetadata)
	case UIMessageChunkTypeError, UIMessageChunkTypeAbort:
		return nil
	default:
		if IsDataUIMessageChunk(chunk) {
			if boolPtrValue(chunk.Transient) {
				return nil
			}
			part := UIPart{Type: chunk.Type, ID: chunk.ID, Data: chunk.Data}
			if chunk.ID != "" {
				for i := range state.Message.Parts {
					if state.Message.Parts[i].Type == chunk.Type && state.Message.Parts[i].ID == chunk.ID {
						state.Message.Parts[i].Data = chunk.Data
						return nil
					}
				}
			}
			state.Message.Parts = append(state.Message.Parts, part)
		}
	}
	return nil
}

func LastAssistantMessageIsCompleteWithApprovalResponses(messages []UIMessage) bool {
	if len(messages) == 0 {
		return false
	}
	message := messages[len(messages)-1]
	if message.Role != RoleAssistant {
		return false
	}
	lastStepStartIndex := -1
	for i, part := range message.Parts {
		if part.Type == "step-start" {
			lastStepStartIndex = i
		}
	}
	hasApprovalResponse := false
	for _, part := range message.Parts[lastStepStartIndex+1:] {
		if !IsToolUIPart(part) {
			continue
		}
		switch part.State {
		case "approval-responded":
			hasApprovalResponse = true
		case "output-available", "output-error":
		default:
			return false
		}
	}
	return hasApprovalResponse
}

func validateUIMessageStreamChunkSchemas(state *StreamingUIMessageState, chunk UIMessageChunk, opts ProcessUIMessageStreamOptions) error {
	if opts.MessageMetadataSchema != nil {
		switch chunk.Type {
		case UIMessageChunkTypeStart, UIMessageChunkTypeFinish, UIMessageChunkTypeMessageMetadata:
			if chunk.MessageMetadata != nil {
				metadata := chunk.MessageMetadata
				if state != nil {
					metadata = mergeUIMessageMetadata(state.Message.Metadata, chunk.MessageMetadata)
				}
				if err := validateUIValueAgainstSchema(opts.MessageMetadataSchema, metadata); err != nil {
					return NewUIMessageStreamError(chunk.Type, chunkIDForValidation(chunk), fmt.Sprintf("message metadata validation failed: %v", err))
				}
			}
		}
	}

	if IsDataUIMessageChunk(chunk) && opts.DataSchemas != nil {
		schema, ok := lookupUIDataSchema(opts.DataSchemas, chunk.Type)
		if !ok {
			return nil
		}
		if err := validateUIValueAgainstSchema(schema, chunk.Data); err != nil {
			return NewUIMessageStreamError(chunk.Type, chunkIDForValidation(chunk), fmt.Sprintf("data validation failed: %v", err))
		}
	}

	return nil
}

type streamingToolPartUpdate struct {
	ToolCallID       string
	ToolName         string
	Dynamic          bool
	State            string
	Input            any
	RawInput         any
	Output           any
	ErrorText        string
	ProviderExecuted *bool
	ProviderMetadata ProviderMetadata
	Preliminary      *bool
	Title            string
}

func firstTextToUIMessageStreamOptions(options []TextToUIMessageStreamOptions) TextToUIMessageStreamOptions {
	if len(options) == 0 {
		return TextToUIMessageStreamOptions{}
	}
	return options[0]
}

func textToUIMessageTextID(opts TextToUIMessageStreamOptions) string {
	if opts.TextID != "" {
		return opts.TextID
	}
	return defaultUITextPartID
}

func writeTextUIMessageStreamStart(writer UIMessageStreamWriter, opts TextToUIMessageStreamOptions) {
	if opts.MessageID != "" {
		writer.Write(StartUIMessageChunk(opts.MessageID))
	} else {
		writer.Write(UIMessageChunk{Type: UIMessageChunkTypeStart})
	}
	writer.Write(UIMessageChunk{Type: UIMessageChunkTypeStartStep})
	writer.Write(TextStartUIMessageChunk(textToUIMessageTextID(opts)))
}

func writeTextUIMessageStreamFinish(writer UIMessageStreamWriter, opts TextToUIMessageStreamOptions) {
	writer.Write(TextEndUIMessageChunk(textToUIMessageTextID(opts)))
	writer.Write(UIMessageChunk{Type: UIMessageChunkTypeFinishStep})
	writer.Write(FinishUIMessageChunk(""))
}

func ensureStreamingUIMessageState(state *StreamingUIMessageState) {
	if state.ActiveTextParts == nil {
		state.ActiveTextParts = map[string]int{}
	}
	if state.ActiveReasoningParts == nil {
		state.ActiveReasoningParts = map[string]int{}
	}
	if state.PartialToolCalls == nil {
		state.PartialToolCalls = map[string]PartialToolCallState{}
	}
	if state.Message.Role == "" {
		state.Message.Role = RoleAssistant
	}
	if state.Message.Parts == nil {
		state.Message.Parts = []UIPart{}
	}
}

func updateStreamingToolPart(state *StreamingUIMessageState, update streamingToolPartUpdate) {
	index, ok := findToolPartIndex(state.Message.Parts, update.ToolCallID)
	if !ok {
		partType := "tool-" + update.ToolName
		if update.Dynamic {
			partType = "dynamic-tool"
		}
		state.Message.Parts = append(state.Message.Parts, UIPart{
			Type:             partType,
			ToolName:         update.ToolName,
			ToolCallID:       update.ToolCallID,
			State:            update.State,
			Input:            update.Input,
			RawInput:         update.RawInput,
			Output:           update.Output,
			ErrorText:        update.ErrorText,
			ProviderExecuted: boolPtrValue(update.ProviderExecuted),
			Preliminary:      boolPtrValue(update.Preliminary),
			Title:            update.Title,
		})
		index = len(state.Message.Parts) - 1
	}

	part := &state.Message.Parts[index]
	part.State = update.State
	if update.Dynamic {
		part.Type = "dynamic-tool"
		part.ToolName = update.ToolName
	}
	part.Input = update.Input
	if update.RawInput != nil {
		part.RawInput = update.RawInput
	}
	part.Output = update.Output
	part.ErrorText = update.ErrorText
	if update.Title != "" {
		part.Title = update.Title
	}
	if update.ProviderExecuted != nil {
		part.ProviderExecuted = *update.ProviderExecuted
	}
	if update.Preliminary != nil {
		part.Preliminary = *update.Preliminary
	}
	if len(update.ProviderMetadata) > 0 {
		if update.State == "output-available" || update.State == "output-error" {
			part.ResultProviderMetadata = mergeMetadata(part.ResultProviderMetadata, update.ProviderMetadata)
		} else {
			part.CallProviderMetadata = mergeMetadata(part.CallProviderMetadata, update.ProviderMetadata)
		}
	}
}

func findToolPartIndex(parts []UIPart, toolCallID string) (int, bool) {
	for i := range parts {
		if IsToolUIPart(parts[i]) && parts[i].ToolCallID == toolCallID {
			return i, true
		}
	}
	return 0, false
}

func getStreamingToolPart(state *StreamingUIMessageState, toolCallID string) (*UIPart, error) {
	index, ok := findToolPartIndex(state.Message.Parts, toolCallID)
	if !ok {
		return nil, NewUIMessageStreamError("tool-invocation", toolCallID, fmt.Sprintf("No tool invocation found for tool call ID %q.", toolCallID))
	}
	return &state.Message.Parts[index], nil
}

func getStreamingToolPartByApprovalID(state *StreamingUIMessageState, approvalID string) (*UIPart, error) {
	for i := range state.Message.Parts {
		part := &state.Message.Parts[i]
		if IsToolUIPart(*part) && part.Approval != nil && part.Approval.ID == approvalID {
			return part, nil
		}
	}
	return nil, NewUIMessageStreamError("tool-approval-response", approvalID, fmt.Sprintf("No tool invocation found for approval ID %q.", approvalID))
}

func parsePragmaticPartialJSON(text string) any {
	if text == "" {
		return nil
	}
	var value any
	if err := json.Unmarshal([]byte(text), &value); err == nil {
		return value
	}
	return text
}

func mergeUIMessageMetadata(existing, incoming any) any {
	if incoming == nil {
		return existing
	}
	if existing == nil {
		return cloneJSONValue(incoming)
	}
	if merged, ok := mergeObjectValues(existing, incoming); ok {
		return merged
	}
	return cloneJSONValue(incoming)
}

func boolPtrValue(value *bool) bool {
	return value != nil && *value
}

func lastAssistantMessage(messages []UIMessage) *UIMessage {
	if len(messages) == 0 {
		return nil
	}
	last := &messages[len(messages)-1]
	if last.Role != RoleAssistant {
		return nil
	}
	return last
}

func isContinuation(originalMessages []UIMessage, response UIMessage) bool {
	last := lastAssistantMessage(originalMessages)
	return last != nil && last.ID == response.ID
}

func messagesWithResponse(originalMessages []UIMessage, response UIMessage) []UIMessage {
	out := cloneUIMessages(originalMessages)
	if isContinuation(out, response) {
		out = out[:len(out)-1]
	}
	return append(out, cloneUIMessage(response))
}

func cloneUIMessage(message UIMessage) UIMessage {
	cloned := cloneUIMessages([]UIMessage{message})
	if len(cloned) == 0 {
		return message
	}
	return cloned[0]
}

func cloneUIMessages(messages []UIMessage) []UIMessage {
	if messages == nil {
		return nil
	}
	data, err := json.Marshal(messages)
	if err != nil {
		return append([]UIMessage(nil), messages...)
	}
	var cloned []UIMessage
	if err := json.Unmarshal(data, &cloned); err != nil {
		return append([]UIMessage(nil), messages...)
	}
	return cloned
}
