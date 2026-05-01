package ai

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

type standardizedPrompt struct {
	System   []Message
	Messages []Message
}

func standardizePrompt(system string, prompt string, messages []Message, allowSystemInMessages bool) (standardizedPrompt, error) {
	if strings.TrimSpace(prompt) == "" && len(messages) == 0 {
		return standardizedPrompt{}, &SDKError{Kind: ErrInvalidPrompt, Message: "prompt or messages must be defined"}
	}
	if strings.TrimSpace(prompt) != "" && len(messages) > 0 {
		return standardizedPrompt{}, &SDKError{Kind: ErrInvalidPrompt, Message: "prompt and messages cannot be defined at the same time"}
	}
	if strings.TrimSpace(prompt) != "" {
		messages = []Message{UserMessage(prompt)}
	}
	if len(messages) == 0 {
		return standardizedPrompt{}, &SDKError{Kind: ErrInvalidPrompt, Message: "messages must not be empty"}
	}
	for _, message := range messages {
		if err := validateModelMessage(message); err != nil {
			return standardizedPrompt{}, err
		}
		if message.Role == RoleSystem && !allowSystemInMessages {
			return standardizedPrompt{}, &SDKError{Kind: ErrInvalidPrompt, Message: "System messages are not allowed in the prompt or messages fields. Use the system option instead."}
		}
	}
	var sys []Message
	if strings.TrimSpace(system) != "" {
		sys = append(sys, SystemMessage(system))
	}
	return standardizedPrompt{System: sys, Messages: messages}, nil
}

type promptConversionOptions struct {
	SupportedURLs map[string][]string
	Download      DownloadFunction
}

func promptConversionOptionsForModel(ctx context.Context, model LanguageModel) (promptConversionOptions, error) {
	if model == nil {
		return promptConversionOptions{}, nil
	}
	supportedURLs, err := model.SupportedURLs(ctx)
	if err != nil {
		return promptConversionOptions{}, err
	}
	return promptConversionOptions{
		SupportedURLs: supportedURLs,
		Download: func(ctx context.Context, rawURL string) ([]byte, string, error) {
			return DownloadURL(ctx, rawURL, nil)
		},
	}, nil
}

func toLanguageModelPrompt(prompt standardizedPrompt, responseMessages []Message) ([]Message, error) {
	return toLanguageModelPromptWithOptions(context.Background(), prompt, responseMessages, promptConversionOptions{})
}

func toLanguageModelPromptWithOptions(ctx context.Context, prompt standardizedPrompt, responseMessages []Message, opts promptConversionOptions) ([]Message, error) {
	out := make([]Message, 0, len(prompt.System)+len(prompt.Messages)+len(responseMessages))
	out = append(out, prompt.System...)
	out = append(out, prompt.Messages...)
	out = append(out, responseMessages...)
	if opts.Download != nil {
		downloaded, err := downloadUnsupportedFileParts(ctx, out, opts)
		if err != nil {
			return nil, err
		}
		out = downloaded
	}

	pendingToolCalls := map[string]struct{}{}
	var combined []Message
	for _, message := range out {
		normalized := normalizeMessage(message)
		if normalized.Role == RoleTool && len(combined) > 0 && combined[len(combined)-1].Role == RoleTool {
			combined[len(combined)-1].Content = append(combined[len(combined)-1].Content, normalized.Content...)
		} else {
			combined = append(combined, normalized)
		}
	}

	for _, message := range combined {
		switch message.Role {
		case RoleAssistant:
			for _, part := range message.Content {
				if call, ok := part.(ToolCallPart); ok && !call.ProviderExecuted {
					pendingToolCalls[call.ToolCallID] = struct{}{}
				}
			}
		case RoleTool:
			for _, part := range message.Content {
				if result, ok := part.(ToolResultPart); ok {
					delete(pendingToolCalls, result.ToolCallID)
				}
			}
		case RoleUser, RoleSystem:
			if len(pendingToolCalls) > 0 {
				return nil, NewMissingToolResultsError(mapKeys(pendingToolCalls))
			}
		}
	}
	if len(pendingToolCalls) > 0 {
		return nil, NewMissingToolResultsError(mapKeys(pendingToolCalls))
	}
	filtered := combined[:0]
	for _, message := range combined {
		if message.Role == RoleTool && len(message.Content) == 0 {
			continue
		}
		filtered = append(filtered, message)
	}
	return filtered, nil
}

func downloadUnsupportedFileParts(ctx context.Context, messages []Message, opts promptConversionOptions) ([]Message, error) {
	download := opts.Download
	if download == nil {
		return messages, nil
	}
	out := make([]Message, len(messages))
	for i, message := range messages {
		converted := message
		if len(message.Content) > 0 {
			converted.Content = make([]Part, len(message.Content))
			for j, part := range message.Content {
				convertedPart, err := downloadUnsupportedFilePart(ctx, part, opts)
				if err != nil {
					return nil, err
				}
				converted.Content[j] = convertedPart
			}
		}
		out[i] = converted
	}
	return out, nil
}

func downloadUnsupportedFilePart(ctx context.Context, part Part, opts promptConversionOptions) (Part, error) {
	switch typed := part.(type) {
	case FilePart:
		data, mediaType, ok, err := downloadUnsupportedFileData(ctx, typed.Data, typed.MediaType, typed.Filename, opts)
		if err != nil || !ok {
			return part, err
		}
		typed.Data = data
		typed.MediaType = mediaType
		return typed, nil
	case ReasoningFilePart:
		data, mediaType, ok, err := downloadUnsupportedFileData(ctx, typed.Data, typed.MediaType, "", opts)
		if err != nil || !ok {
			return part, err
		}
		typed.Data = data
		typed.MediaType = mediaType
		return typed, nil
	default:
		return part, nil
	}
}

func downloadUnsupportedFileData(ctx context.Context, data FileData, mediaType string, filename string, opts promptConversionOptions) (FileData, string, bool, error) {
	if data.Type != FileDataTypeURL || data.URL == "" || IsURLSupported(opts.SupportedURLs, mediaType, data.URL) {
		return data, mediaType, false, nil
	}
	downloaded, downloadedMediaType, err := opts.Download(ctx, data.URL)
	if err != nil {
		return FileData{}, "", false, err
	}
	if mediaType == "" {
		mediaType = normalizeMediaType(downloadedMediaType)
	}
	if mediaType == "" {
		mediaType = DetectMediaType(downloaded, filename)
	}
	return FileData{Type: FileDataTypeBytes, Data: cloneBytes(downloaded)}, mediaType, true, nil
}

func normalizeMessage(message Message) Message {
	if len(message.Content) == 0 && message.Text != "" {
		switch message.Role {
		case RoleSystem:
			message.Content = nil
		default:
			message.Content = []Part{TextPart{Text: message.Text}}
		}
	}
	if message.Role == RoleSystem && message.Text == "" {
		for _, part := range message.Content {
			if text, ok := part.(TextPart); ok {
				message.Text += text.Text
			}
		}
	}
	content := message.Content[:0]
	for _, part := range message.Content {
		if text, ok := part.(TextPart); ok && text.Text == "" && text.ProviderOptions == nil {
			continue
		}
		content = append(content, part)
	}
	message.Content = content
	return message
}

func validateModelMessage(message Message) error {
	switch message.Role {
	case RoleSystem:
		if len(message.Content) > 0 {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "system message content must be text"}
		}
	case RoleUser:
		for _, part := range message.Content {
			switch typed := part.(type) {
			case TextPart:
			case FilePart:
				if err := validateFilePart(typed, false); err != nil {
					return err
				}
			default:
				return &SDKError{Kind: ErrInvalidPrompt, Message: fmt.Sprintf("user message contains unsupported %q part", part.PartType())}
			}
		}
	case RoleAssistant:
		for _, part := range message.Content {
			switch typed := part.(type) {
			case TextPart, ReasoningPart, ToolCallPart, ToolResultPart:
			case FilePart:
				if err := validateFilePart(typed, false); err != nil {
					return err
				}
			case ReasoningFilePart:
				if err := validateReasoningFilePart(typed); err != nil {
					return err
				}
			default:
				return &SDKError{Kind: ErrInvalidPrompt, Message: fmt.Sprintf("assistant message contains unsupported %q part", part.PartType())}
			}
		}
	case RoleTool:
		for _, part := range message.Content {
			if _, ok := part.(ToolResultPart); !ok {
				return &SDKError{Kind: ErrInvalidPrompt, Message: fmt.Sprintf("tool message contains unsupported %q part", part.PartType())}
			}
		}
	case "":
		return &SDKError{Kind: ErrInvalidPrompt, Message: "message role is required"}
	default:
		return NewInvalidMessageRoleError(message.Role)
	}
	return nil
}

func validateFilePart(part FilePart, allowMissingMediaType bool) error {
	if !allowMissingMediaType && strings.TrimSpace(part.MediaType) == "" {
		return &SDKError{Kind: ErrInvalidPrompt, Message: "file part media type is required"}
	}
	return validateFileData(part.Data, true)
}

func validateReasoningFilePart(part ReasoningFilePart) error {
	if strings.TrimSpace(part.MediaType) == "" {
		return &SDKError{Kind: ErrInvalidPrompt, Message: "reasoning-file part media type is required"}
	}
	if err := validateFileData(part.Data, false); err != nil {
		return err
	}
	if part.Data.Type == FileDataTypeReference || part.Data.Type == FileDataTypeText {
		return &SDKError{Kind: ErrInvalidPrompt, Message: "reasoning-file data must be bytes or url"}
	}
	return nil
}

func validateFileData(data FileData, allowReferenceAndText bool) error {
	switch data.Type {
	case FileDataTypeBytes:
		if len(data.Data) == 0 {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data bytes must not be empty"}
		}
	case FileDataTypeURL:
		if strings.TrimSpace(data.URL) == "" {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data url is required"}
		}
	case FileDataTypeText:
		if !allowReferenceAndText {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data text is not supported here"}
		}
		if data.Text == "" {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data text must not be empty"}
		}
	case FileDataTypeReference:
		if !allowReferenceAndText {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data reference is not supported here"}
		}
		if strings.TrimSpace(data.Reference) == "" && len(data.ProviderReference) == 0 {
			return &SDKError{Kind: ErrInvalidPrompt, Message: "file data reference is required"}
		}
	default:
		return &SDKError{Kind: ErrInvalidPrompt, Message: fmt.Sprintf("unsupported file data type %q", data.Type)}
	}
	return nil
}

func mapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
