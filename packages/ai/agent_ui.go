package ai

import (
	"context"
	"encoding/json"
)

type AgentUIStreamOptions struct {
	Agent     Agent
	Call      AgentStreamOptions
	MessageID string
	TextID    string
}

func CreateAgentUIStream(ctx context.Context, opts AgentUIStreamOptions) <-chan UIMessageChunk {
	return CreateUIMessageStream(CreateUIMessageStreamOptions{
		Execute: func(writer UIMessageStreamWriter) error {
			result, err := opts.Agent.Stream(ctx, opts.Call)
			if err != nil {
				return err
			}
			return writeStreamTextResultAsUIMessageChunks(ctx, writer, result, StreamTextUIMessageStreamOptions{
				MessageID: opts.MessageID,
				TextID:    opts.TextID,
			})
		},
	})
}

type StreamTextUIMessageStreamOptions struct {
	MessageID  string
	TextID     string
	BufferSize int
}

func CreateStreamTextUIMessageStream(ctx context.Context, result *StreamTextResult, options ...StreamTextUIMessageStreamOptions) <-chan UIMessageChunk {
	opts := firstStreamTextUIMessageStreamOptions(options)
	return CreateUIMessageStream(CreateUIMessageStreamOptions{
		BufferSize: opts.BufferSize,
		Execute: func(writer UIMessageStreamWriter) error {
			return writeStreamTextResultAsUIMessageChunks(ctx, writer, result, opts)
		},
	})
}

func writeStreamTextResultAsUIMessageChunks(ctx context.Context, writer UIMessageStreamWriter, result *StreamTextResult, opts StreamTextUIMessageStreamOptions) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if result == nil || result.Stream == nil {
		return &SDKError{Kind: ErrNoOutputGenerated, Message: "stream text result is empty"}
	}
	messageID := opts.MessageID
	if messageID == "" {
		messageID = "message-1"
	}
	textID := opts.TextID
	if textID == "" {
		textID = "text-1"
	}
	writer.Write(StartUIMessageChunk(messageID))
	textStarted := false
	reasoningStarted := map[string]bool{}
	for {
		var part StreamPart
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case part, ok = <-result.Stream:
			if !ok {
				return nil
			}
		}
		switch part.Type {
		case "text-delta":
			if !textStarted {
				writer.Write(TextStartUIMessageChunk(textID))
				textStarted = true
			}
			writer.Write(TextDeltaUIMessageChunk(textID, part.TextDelta))
		case "reasoning-delta":
			id := part.ID
			if id == "" {
				id = "reasoning-1"
			}
			if !reasoningStarted[id] {
				writer.Write(UIMessageChunk{Type: UIMessageChunkTypeReasoningStart, ID: id, ProviderMetadata: part.ProviderMetadata})
				reasoningStarted[id] = true
			}
			writer.Write(UIMessageChunk{Type: UIMessageChunkTypeReasoningDelta, ID: id, Delta: part.ReasoningDelta, ProviderMetadata: part.ProviderMetadata})
		case "tool-input-delta":
			writer.Write(UIMessageChunk{Type: UIMessageChunkTypeToolInputDelta, ToolCallID: part.ToolCallID, InputTextDelta: part.ToolInputDelta})
		case "tool-call":
			input, _ := parseUIStreamToolInput(part.ToolInput)
			writer.Write(UIMessageChunk{
				Type:             UIMessageChunkTypeToolInputAvailable,
				ToolCallID:       part.ToolCallID,
				ToolName:         part.ToolName,
				Input:            input,
				ProviderMetadata: part.ProviderMetadata,
			})
		case "tool-result":
			output := any(part.Content)
			if resultPart, ok := part.Content.(ToolResultPart); ok {
				output = resultPart.Output.Value
				if resultPart.Output.Type == "execution-denied" {
					writer.Write(UIMessageChunk{Type: UIMessageChunkTypeToolOutputDenied, ToolCallID: part.ToolCallID})
					continue
				}
				if resultPart.Output.Type == "error-text" || resultPart.Output.Type == "error-json" {
					writer.Write(UIMessageChunk{Type: UIMessageChunkTypeToolOutputError, ToolCallID: part.ToolCallID, ErrorText: stringifyToolOutput(resultPart.Output.Value)})
					continue
				}
			}
			writer.Write(UIMessageChunk{Type: UIMessageChunkTypeToolOutputAvailable, ToolCallID: part.ToolCallID, Output: output, ProviderMetadata: part.ProviderMetadata})
		case "file":
			if file, ok := part.Content.(FilePart); ok {
				writer.Write(UIMessageChunk{Type: UIMessageChunkTypeFile, URL: file.Data.URL, MediaType: file.MediaType, Filename: file.Filename, ProviderMetadata: ProviderMetadata(file.ProviderOptions)})
			}
		case "source":
			chunk := UIMessageChunk{Type: UIMessageChunkTypeSourceURL, SourceID: part.ID, ProviderMetadata: part.ProviderMetadata}
			if source, ok := part.Content.(SourcePart); ok {
				chunk.SourceID = source.ID
				chunk.URL = source.URL
				chunk.Title = source.Title
			}
			writer.Write(chunk)
		case "finish-step":
			writer.Write(UIMessageChunk{Type: UIMessageChunkTypeFinishStep})
		case "finish":
			if textStarted {
				writer.Write(TextEndUIMessageChunk(textID))
			}
			for id := range reasoningStarted {
				writer.Write(UIMessageChunk{Type: UIMessageChunkTypeReasoningEnd, ID: id})
			}
			writer.Write(FinishUIMessageChunk(part.FinishReason.Unified))
		case "error":
			writer.Write(ErrorUIMessageChunk(part.Err))
		default:
			if part.Type == "" || part.Type == "raw" {
				continue
			}
			writer.Write(UIMessageChunk{Type: UIMessageChunkTypeCustom, Kind: part.Type, ProviderMetadata: part.ProviderMetadata})
		}
	}
}

func firstStreamTextUIMessageStreamOptions(options []StreamTextUIMessageStreamOptions) StreamTextUIMessageStreamOptions {
	if len(options) == 0 {
		return StreamTextUIMessageStreamOptions{}
	}
	return options[0]
}

func stringifyToolOutput(value any) string {
	if value == nil {
		return ""
	}
	if s, ok := value.(string); ok {
		return s
	}
	bytes, err := json.Marshal(value)
	if err != nil {
		return "tool execution failed"
	}
	return string(bytes)
}

func parseUIStreamToolInput(input string) (any, error) {
	if input == "" {
		return nil, nil
	}
	var out any
	if err := json.Unmarshal([]byte(input), &out); err != nil {
		return input, err
	}
	return out, nil
}
