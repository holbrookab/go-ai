package ai

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var ErrInvalidUIMessage = errors.New("invalid UI message")

type UIMessage struct {
	ID       string   `json:"id"`
	Role     Role     `json:"role"`
	Metadata any      `json:"metadata,omitempty"`
	Parts    []UIPart `json:"parts"`
}

type UIPart struct {
	Type string `json:"type"`

	Text  string `json:"text,omitempty"`
	State string `json:"state,omitempty"`
	Kind  string `json:"kind,omitempty"`

	MediaType         string `json:"mediaType,omitempty"`
	Filename          string `json:"filename,omitempty"`
	URL               string `json:"url,omitempty"`
	ProviderReference any    `json:"providerReference,omitempty"`

	ID       string `json:"id,omitempty"`
	Data     any    `json:"data,omitempty"`
	SourceID string `json:"sourceId,omitempty"`
	Title    string `json:"title,omitempty"`

	ToolName         string `json:"toolName,omitempty"`
	ToolCallID       string `json:"toolCallId,omitempty"`
	Input            any    `json:"input,omitempty"`
	RawInput         any    `json:"rawInput,omitempty"`
	Output           any    `json:"output,omitempty"`
	ErrorText        string `json:"errorText,omitempty"`
	ProviderExecuted bool   `json:"providerExecuted,omitempty"`
	Dynamic          bool   `json:"dynamic,omitempty"`
	Preliminary      bool   `json:"preliminary,omitempty"`
	Approval         *struct {
		ID          string `json:"id"`
		Approved    *bool  `json:"approved,omitempty"`
		Reason      string `json:"reason,omitempty"`
		IsAutomatic bool   `json:"isAutomatic,omitempty"`
	} `json:"approval,omitempty"`

	ProviderMetadata       ProviderMetadata `json:"providerMetadata,omitempty"`
	CallProviderMetadata   ProviderMetadata `json:"callProviderMetadata,omitempty"`
	ResultProviderMetadata ProviderMetadata `json:"resultProviderMetadata,omitempty"`
}

func IsDataUIPart(part UIPart) bool {
	return strings.HasPrefix(part.Type, "data-")
}

func IsTextUIPart(part UIPart) bool {
	return part.Type == "text"
}

func IsFileUIPart(part UIPart) bool {
	return part.Type == "file"
}

func IsReasoningUIPart(part UIPart) bool {
	return part.Type == "reasoning"
}

func IsReasoningFileUIPart(part UIPart) bool {
	return part.Type == "reasoning-file"
}

func IsDynamicToolUIPart(part UIPart) bool {
	return part.Type == "dynamic-tool"
}

func IsStaticToolUIPart(part UIPart) bool {
	return strings.HasPrefix(part.Type, "tool-") && part.Type != "tool-"
}

func IsToolUIPart(part UIPart) bool {
	return IsStaticToolUIPart(part) || IsDynamicToolUIPart(part)
}

func StaticToolName(part UIPart) string {
	if !IsStaticToolUIPart(part) {
		return ""
	}
	return strings.TrimPrefix(part.Type, "tool-")
}

func ToolName(part UIPart) string {
	if IsDynamicToolUIPart(part) {
		return part.ToolName
	}
	return StaticToolName(part)
}

type UIValueValidator func(any) error

type ValidateUIMessagesOptions struct {
	MetadataSchema any
	DataSchemas    map[string]any
	Tools          map[string]Tool
}

type SafeValidateUIMessagesResult struct {
	Success  bool
	Messages []UIMessage
	Error    error
}

func SafeValidateUIMessages(messages []UIMessage, options ...ValidateUIMessagesOptions) SafeValidateUIMessagesResult {
	if err := ValidateUIMessages(messages, options...); err != nil {
		return SafeValidateUIMessagesResult{Success: false, Error: err}
	}
	return SafeValidateUIMessagesResult{Success: true, Messages: messages}
}

func ValidateUIMessages(messages []UIMessage, options ...ValidateUIMessagesOptions) error {
	if len(messages) == 0 {
		return &SDKError{Kind: ErrInvalidUIMessage, Message: "messages must not be empty"}
	}
	opts := firstValidateUIMessagesOptions(options)
	for i, message := range messages {
		if message.ID == "" {
			return invalidUIMessagef("messages[%d].id is required", i)
		}
		switch message.Role {
		case RoleSystem, RoleUser, RoleAssistant:
		default:
			return invalidUIMessagef("messages[%d].role must be system, user, or assistant", i)
		}
		if len(message.Parts) == 0 {
			return invalidUIMessagef("messages[%d].parts must not be empty", i)
		}
		for j, part := range message.Parts {
			if err := validateUIPart(part); err != nil {
				return invalidUIMessagef("messages[%d].parts[%d]: %v", i, j, err)
			}
			if err := validateUIPartWithSchemas(i, j, part, opts); err != nil {
				return err
			}
		}
		if opts.MetadataSchema != nil {
			if err := validateUIValueAgainstSchema(opts.MetadataSchema, message.Metadata); err != nil {
				return invalidUIMessagef("messages[%d].metadata (id: %q) validation failed: %v", i, message.ID, err)
			}
		}
	}
	return nil
}

type ConvertToModelMessagesOptions struct {
	Tools                     map[string]Tool
	IgnoreIncompleteToolCalls bool
	ConvertDataPart           func(UIPart) (Part, bool, error)
}

func ConvertToModelMessages(messages []UIMessage, opts ConvertToModelMessagesOptions) ([]Message, error) {
	if err := ValidateUIMessages(messages, ValidateUIMessagesOptions{Tools: opts.Tools}); err != nil {
		return nil, err
	}

	modelMessages := make([]Message, 0, len(messages))
	for _, message := range messages {
		switch message.Role {
		case RoleSystem:
			var text strings.Builder
			providerOptions := ProviderOptions{}
			for _, part := range message.Parts {
				if IsTextUIPart(part) {
					text.WriteString(part.Text)
					providerOptions = mergeProviderOptions(providerOptions, ProviderOptions(part.ProviderMetadata))
				}
			}
			modelMessages = append(modelMessages, Message{
				Role:            RoleSystem,
				Text:            text.String(),
				ProviderOptions: nilIfEmptyProviderOptions(providerOptions),
			})

		case RoleUser:
			content, err := convertUserUIParts(message.Parts, opts)
			if err != nil {
				return nil, err
			}
			modelMessages = append(modelMessages, Message{Role: RoleUser, Content: content})

		case RoleAssistant:
			converted, err := convertAssistantUIMessage(message, opts)
			if err != nil {
				return nil, err
			}
			modelMessages = append(modelMessages, converted...)
		}
	}
	return modelMessages, nil
}

func AppendResponseMessages(messages []UIMessage, responseMessages []Message) []UIMessage {
	out := append([]UIMessage(nil), messages...)
	for _, message := range responseMessages {
		switch message.Role {
		case RoleAssistant:
			parts := partsToUIParts(message.Content)
			if message.Text != "" && len(parts) == 0 {
				parts = append(parts, UIPart{Type: "text", Text: message.Text})
			}
			if len(out) > 0 && out[len(out)-1].Role == RoleAssistant {
				out[len(out)-1].Parts = append(out[len(out)-1].Parts, parts...)
				continue
			}
			out = append(out, UIMessage{
				ID:    fmt.Sprintf("response-%d", len(out)+1),
				Role:  RoleAssistant,
				Parts: parts,
			})
		case RoleTool:
			if len(out) == 0 || out[len(out)-1].Role != RoleAssistant {
				out = append(out, UIMessage{
					ID:    fmt.Sprintf("response-%d", len(out)+1),
					Role:  RoleAssistant,
					Parts: partsToUIParts(message.Content),
				})
				continue
			}
			applyToolResultsToUIMessage(&out[len(out)-1], message.Content)
		}
	}
	return out
}

func validateUIPart(part UIPart) error {
	if part.Type == "" {
		return errors.New("type is required")
	}
	switch {
	case IsTextUIPart(part), IsReasoningUIPart(part):
		if part.State != "" && part.State != "streaming" && part.State != "done" {
			return fmt.Errorf("state %q is not supported", part.State)
		}
	case IsFileUIPart(part), IsReasoningFileUIPart(part):
		if part.MediaType == "" {
			return errors.New("mediaType is required")
		}
		if part.URL == "" && part.ProviderReference == nil {
			return errors.New("url or providerReference is required")
		}
	case IsDataUIPart(part):
	case part.Type == "step-start":
	case part.Type == "source-url":
		if part.SourceID == "" || part.URL == "" {
			return errors.New("sourceId and url are required")
		}
	case part.Type == "source-document":
		if part.SourceID == "" || part.MediaType == "" || part.Title == "" {
			return errors.New("sourceId, mediaType, and title are required")
		}
	case part.Type == "custom":
		if part.Kind == "" {
			return errors.New("kind is required")
		}
	case IsToolUIPart(part):
		if part.ToolCallID == "" {
			return errors.New("toolCallId is required")
		}
		if IsDynamicToolUIPart(part) && part.ToolName == "" {
			return errors.New("toolName is required")
		}
		switch part.State {
		case "input-streaming", "input-available", "approval-requested", "approval-responded", "output-available", "output-error", "output-denied":
		default:
			return fmt.Errorf("state %q is not supported", part.State)
		}
		if strings.HasPrefix(part.State, "approval-") || part.State == "output-denied" {
			if part.Approval == nil || part.Approval.ID == "" {
				return errors.New("approval.id is required")
			}
		}
		if part.State == "output-error" && part.ErrorText == "" {
			return errors.New("errorText is required")
		}
	default:
		return fmt.Errorf("unsupported type %q", part.Type)
	}
	return nil
}

func convertUserUIParts(parts []UIPart, opts ConvertToModelMessagesOptions) ([]Part, error) {
	content := make([]Part, 0, len(parts))
	for _, part := range parts {
		switch {
		case IsTextUIPart(part):
			content = append(content, TextPart{Text: part.Text, ProviderMetadata: part.ProviderMetadata, ProviderOptions: ProviderOptions(part.ProviderMetadata)})
		case IsFileUIPart(part):
			content = append(content, fileUIPartToModelPart(part))
		case IsDataUIPart(part) && opts.ConvertDataPart != nil:
			converted, ok, err := opts.ConvertDataPart(part)
			if err != nil {
				return nil, err
			}
			if ok {
				content = append(content, converted)
			}
		}
	}
	return content, nil
}

func convertAssistantUIMessage(message UIMessage, opts ConvertToModelMessagesOptions) ([]Message, error) {
	var out []Message
	var block []UIPart
	processBlock := func() error {
		if len(block) == 0 {
			return nil
		}
		content := make([]Part, 0, len(block))
		toolParts := make([]UIPart, 0)
		for _, part := range block {
			switch {
			case IsTextUIPart(part):
				content = append(content, TextPart{Text: part.Text, ProviderMetadata: part.ProviderMetadata, ProviderOptions: ProviderOptions(part.ProviderMetadata)})
			case IsReasoningUIPart(part):
				content = append(content, ReasoningPart{Text: part.Text, ProviderOptions: ProviderOptions(part.ProviderMetadata), ProviderMetadata: part.ProviderMetadata})
			case IsReasoningFileUIPart(part):
				content = append(content, ReasoningFilePart{Data: fileDataFromUIPart(part), MediaType: part.MediaType, ProviderOptions: ProviderOptions(part.ProviderMetadata), ProviderMetadata: part.ProviderMetadata})
			case IsFileUIPart(part):
				content = append(content, fileUIPartToModelPart(part))
			case IsToolUIPart(part):
				if opts.IgnoreIncompleteToolCalls && (part.State == "input-streaming" || part.State == "input-available") {
					continue
				}
				if part.State != "input-streaming" {
					content = append(content, toolCallUIPartToModelPart(part))
					if part.ProviderExecuted && (part.State == "output-available" || part.State == "output-error") {
						result, err := toolResultUIPartToModelPart(part, opts, true)
						if err != nil {
							return err
						}
						content = append(content, result)
					}
				}
				if part.ProviderExecuted || part.Approval != nil {
					toolParts = append(toolParts, part)
				} else if part.State == "output-available" || part.State == "output-error" || part.State == "output-denied" {
					toolParts = append(toolParts, part)
				}
			case IsDataUIPart(part) && opts.ConvertDataPart != nil:
				converted, ok, err := opts.ConvertDataPart(part)
				if err != nil {
					return err
				}
				if ok {
					content = append(content, converted)
				}
			}
		}
		if len(content) > 0 {
			out = append(out, Message{Role: RoleAssistant, Content: content})
		}
		toolContent, err := toolResultContentFromUIParts(toolParts, opts)
		if err != nil {
			return err
		}
		if len(toolContent) > 0 {
			out = append(out, Message{Role: RoleTool, Content: toolContent})
		}
		block = block[:0]
		return nil
	}

	for _, part := range message.Parts {
		if opts.IgnoreIncompleteToolCalls && IsToolUIPart(part) && (part.State == "input-streaming" || part.State == "input-available") {
			continue
		}
		if part.Type == "step-start" {
			if err := processBlock(); err != nil {
				return nil, err
			}
			continue
		}
		block = append(block, part)
	}
	if err := processBlock(); err != nil {
		return nil, err
	}
	return out, nil
}

func toolResultContentFromUIParts(parts []UIPart, opts ConvertToModelMessagesOptions) ([]Part, error) {
	content := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Approval != nil && part.Approval.Approved != nil && !*part.Approval.Approved && part.State == "approval-responded" {
			content = append(content, ToolResultPart{
				ToolCallID:       part.ToolCallID,
				ToolName:         ToolName(part),
				Input:            part.Input,
				Output:           ToolResultOutput{Type: "execution-denied", Reason: part.Approval.Reason},
				ProviderExecuted: part.ProviderExecuted,
				Dynamic:          IsDynamicToolUIPart(part),
				ProviderOptions:  ProviderOptions(part.CallProviderMetadata),
				ProviderMetadata: part.CallProviderMetadata,
			})
		}
		if part.ProviderExecuted {
			continue
		}
		switch part.State {
		case "output-denied":
			reason := "Tool call execution denied."
			if part.Approval != nil && part.Approval.Reason != "" {
				reason = part.Approval.Reason
			}
			content = append(content, ToolResultPart{
				ToolCallID:       part.ToolCallID,
				ToolName:         ToolName(part),
				Input:            part.Input,
				Output:           ToolResultOutput{Type: "error-text", Value: reason},
				ProviderExecuted: part.ProviderExecuted,
				Dynamic:          IsDynamicToolUIPart(part),
				ProviderOptions:  ProviderOptions(part.CallProviderMetadata),
				ProviderMetadata: part.CallProviderMetadata,
			})
		case "output-error", "output-available":
			result, err := toolResultUIPartToModelPart(part, opts, false)
			if err != nil {
				return nil, err
			}
			content = append(content, result)
		}
	}
	return content, nil
}

func toolResultUIPartToModelPart(part UIPart, opts ConvertToModelMessagesOptions, providerExecutedResult bool) (ToolResultPart, error) {
	toolName := ToolName(part)
	outputValue := part.Output
	isErr := part.State == "output-error"
	if isErr {
		outputValue = part.ErrorText
	}
	modelOutput, err := CreateToolModelOutput(opts.Tools[toolName], part.ToolCallID, part.Input, outputValue, isErr)
	if err != nil {
		return ToolResultPart{}, err
	}
	providerMetadata := part.ResultProviderMetadata
	if providerExecutedResult && len(providerMetadata) == 0 {
		providerMetadata = part.CallProviderMetadata
	}
	return ToolResultPart{
		ToolCallID:       part.ToolCallID,
		ToolName:         toolName,
		Input:            part.Input,
		Output:           modelOutput,
		Result:           part.Output,
		IsError:          isErr,
		ProviderExecuted: part.ProviderExecuted,
		Dynamic:          IsDynamicToolUIPart(part),
		Preliminary:      part.Preliminary,
		ProviderOptions:  ProviderOptions(providerMetadata),
		ProviderMetadata: providerMetadata,
	}, nil
}

func toolCallUIPartToModelPart(part UIPart) ToolCallPart {
	input := part.Input
	if part.State == "output-error" && input == nil {
		input = part.RawInput
	}
	return ToolCallPart{
		ToolCallID:       part.ToolCallID,
		ToolName:         ToolName(part),
		Input:            input,
		ProviderExecuted: part.ProviderExecuted,
		Dynamic:          IsDynamicToolUIPart(part),
		Title:            part.Title,
		ProviderOptions:  ProviderOptions(part.CallProviderMetadata),
		ProviderMetadata: part.CallProviderMetadata,
	}
}

func fileUIPartToModelPart(part UIPart) FilePart {
	return FilePart{
		Data:             fileDataFromUIPart(part),
		MediaType:        part.MediaType,
		Filename:         part.Filename,
		ProviderMetadata: part.ProviderMetadata,
		ProviderOptions:  ProviderOptions(part.ProviderMetadata),
	}
}

func fileDataFromUIPart(part UIPart) FileData {
	if part.ProviderReference != nil {
		return FileData{Type: "reference", Reference: providerReferenceString(part.ProviderReference)}
	}
	return FileData{Type: "url", URL: part.URL}
}

func providerReferenceString(reference any) string {
	if s, ok := reference.(string); ok {
		return s
	}
	b, err := json.Marshal(reference)
	if err != nil {
		return fmt.Sprint(reference)
	}
	return string(b)
}

func partsToUIParts(parts []Part) []UIPart {
	out := make([]UIPart, 0, len(parts))
	for _, part := range parts {
		switch part := part.(type) {
		case TextPart:
			out = append(out, UIPart{Type: "text", Text: part.Text, State: "done", ProviderMetadata: mergeMetadata(ProviderMetadata(part.ProviderOptions), part.ProviderMetadata)})
		case FilePart:
			uiPart := UIPart{Type: "file", MediaType: part.MediaType, Filename: part.Filename, URL: part.Data.URL, ProviderMetadata: mergeMetadata(ProviderMetadata(part.ProviderOptions), part.ProviderMetadata)}
			if part.Data.Type == "reference" {
				uiPart.ProviderReference = part.Data.Reference
			}
			out = append(out, uiPart)
		case ReasoningPart:
			out = append(out, UIPart{Type: "reasoning", Text: part.Text, State: "done", ProviderMetadata: part.ProviderMetadata})
		case ReasoningFilePart:
			out = append(out, UIPart{Type: "reasoning-file", MediaType: part.MediaType, URL: part.Data.URL, ProviderMetadata: part.ProviderMetadata})
		case ToolCallPart:
			partType := "tool-" + part.ToolName
			if part.Dynamic {
				partType = "dynamic-tool"
			}
			out = append(out, UIPart{Type: partType, ToolName: part.ToolName, ToolCallID: part.ToolCallID, State: "input-available", Input: part.Input, ProviderExecuted: part.ProviderExecuted, Dynamic: part.Dynamic, Title: part.Title, CallProviderMetadata: part.ProviderMetadata})
		case ToolResultPart:
			partType := "tool-" + part.ToolName
			if part.Dynamic {
				partType = "dynamic-tool"
			}
			out = append(out, UIPart{Type: partType, ToolName: part.ToolName, ToolCallID: part.ToolCallID, State: "output-available", Input: part.Input, Output: part.Result, ProviderExecuted: part.ProviderExecuted, Dynamic: part.Dynamic, Preliminary: part.Preliminary, ResultProviderMetadata: part.ProviderMetadata})
		}
	}
	return out
}

func applyToolResultsToUIMessage(message *UIMessage, parts []Part) {
	for _, part := range parts {
		result, ok := part.(ToolResultPart)
		if !ok {
			continue
		}
		found := false
		for i := range message.Parts {
			if message.Parts[i].ToolCallID != result.ToolCallID {
				continue
			}
			message.Parts[i].State = "output-available"
			if result.IsError {
				message.Parts[i].State = "output-error"
				if s, ok := result.Output.Value.(string); ok {
					message.Parts[i].ErrorText = s
				}
			}
			message.Parts[i].Output = result.Result
			message.Parts[i].ResultProviderMetadata = result.ProviderMetadata
			found = true
			break
		}
		if !found {
			message.Parts = append(message.Parts, partsToUIParts([]Part{result})...)
		}
	}
}

func invalidUIMessagef(format string, args ...any) error {
	return &SDKError{Kind: ErrInvalidUIMessage, Message: fmt.Sprintf(format, args...)}
}

func firstValidateUIMessagesOptions(options []ValidateUIMessagesOptions) ValidateUIMessagesOptions {
	if len(options) == 0 {
		return ValidateUIMessagesOptions{}
	}
	return options[0]
}

func validateUIPartWithSchemas(messageIndex, partIndex int, part UIPart, opts ValidateUIMessagesOptions) error {
	if IsDataUIPart(part) && opts.DataSchemas != nil {
		dataName := strings.TrimPrefix(part.Type, "data-")
		schema, ok := lookupUIDataSchema(opts.DataSchemas, part.Type)
		if !ok {
			return invalidUIMessagef(
				"messages[%d].parts[%d].data (id: %q, name: %q) validation failed: no data schema found",
				messageIndex,
				partIndex,
				part.ID,
				dataName,
			)
		}
		if err := validateUIValueAgainstSchema(schema, part.Data); err != nil {
			return invalidUIMessagef(
				"messages[%d].parts[%d].data (id: %q, name: %q) validation failed: %v",
				messageIndex,
				partIndex,
				part.ID,
				dataName,
				err,
			)
		}
	}

	if opts.Tools != nil && IsToolUIPart(part) {
		toolName := ToolName(part)
		tool, ok := opts.Tools[toolName]
		if !ok && (part.State == "output-available" || part.State == "output-error" || part.State == "output-denied") {
			return nil
		}
		if !ok {
			return invalidUIMessagef(
				"messages[%d].parts[%d].input (toolCallId: %q, toolName: %q) validation failed: no tool schema found",
				messageIndex,
				partIndex,
				part.ToolCallID,
				toolName,
			)
		}
		shouldValidateInput := part.State == "input-available" || part.State == "output-available" || (part.State == "output-error" && part.Input != nil)
		if shouldValidateInput {
			if err := validateUIToolInput(tool, part.Input); err != nil {
				return invalidUIMessagef(
					"messages[%d].parts[%d].input (toolCallId: %q, toolName: %q) validation failed: %v",
					messageIndex,
					partIndex,
					part.ToolCallID,
					toolName,
					err,
				)
			}
		}
		if part.State == "output-available" {
			if err := validateUIToolOutput(tool, part.Output); err != nil {
				return invalidUIMessagef(
					"messages[%d].parts[%d].output (toolCallId: %q, toolName: %q) validation failed: %v",
					messageIndex,
					partIndex,
					part.ToolCallID,
					toolName,
					err,
				)
			}
		}
	}
	return nil
}

func validateUIToolInput(tool Tool, input any) error {
	return ValidateToolInput(tool, input)
}

func validateUIToolOutput(tool Tool, output any) error {
	return ValidateToolOutput(tool, output)
}

func lookupUIDataSchema(schemas map[string]any, partType string) (any, bool) {
	if schemas == nil {
		return nil, false
	}
	dataName := strings.TrimPrefix(partType, "data-")
	if schema, ok := schemas[dataName]; ok {
		return schema, true
	}
	schema, ok := schemas[partType]
	return schema, ok
}

func validateUIValueAgainstSchema(schema any, value any) error {
	switch validator := schema.(type) {
	case nil:
		return nil
	case UIValueValidator:
		return validator(value)
	case func(any) error:
		return validator(value)
	default:
		return validateJSONSchema(normalizeSchema(schema), value, "$")
	}
}

func nilIfEmptyProviderOptions(options ProviderOptions) ProviderOptions {
	if len(options) == 0 {
		return nil
	}
	return options
}
