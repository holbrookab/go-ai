package ai

import (
	"context"
	"encoding/json"
)

type ToolCall struct {
	ToolCallID       string
	ToolName         string
	Input            any
	ProviderExecuted bool
	Dynamic          bool
	Invalid          bool
	Error            error
	ProviderMetadata ProviderMetadata
}

type ToolExecutionOptions struct {
	ToolCallID string
	Messages   []Message
	Context    any
}

type Tool struct {
	Name             string
	Title            string
	Description      string
	InputSchema      any
	OutputSchema     any
	InputExamples    []any
	Strict           *bool
	ProviderOptions  ProviderOptions
	ProviderMetadata ProviderMetadata
	Type             string
	ID               string
	Args             any
	Execute          func(context.Context, ToolCall, ToolExecutionOptions) (any, error)
	ValidateInput    func(any) error
	ValidateOutput   func(any) error
	ToModelOutput    func(toolCallID string, input any, output any) (ToolResultOutput, error)
	NeedsApproval    func(context.Context, ToolCall) (ApprovalDecision, error)
}

type ApprovalDecision struct {
	Type   string
	Reason string
}

func Approved(reason string) ApprovalDecision {
	return ApprovalDecision{Type: "approved", Reason: reason}
}

func Denied(reason string) ApprovalDecision {
	return ApprovalDecision{Type: "denied", Reason: reason}
}

func UserApproval() ApprovalDecision {
	return ApprovalDecision{Type: "user-approval"}
}

func (t Tool) toModelTool(name string) ModelTool {
	toolType := t.Type
	if toolType == "" || toolType == "function" || toolType == "dynamic" {
		return ModelTool{
			Type:            "function",
			Name:            name,
			Description:     t.Description,
			InputSchema:     normalizeSchema(t.InputSchema),
			InputExamples:   t.InputExamples,
			Strict:          t.Strict,
			ProviderOptions: t.ProviderOptions,
		}
	}
	return ModelTool{
		Type: "provider",
		Name: name,
		ID:   t.ID,
		Args: t.Args,
	}
}

func normalizeSchema(schema any) any {
	switch value := schema.(type) {
	case nil:
		return map[string]any{"type": "object", "properties": map[string]any{}}
	case json.RawMessage:
		var out any
		if err := json.Unmarshal(value, &out); err == nil {
			return out
		}
		return value
	default:
		return value
	}
}

func CreateToolModelOutput(tool Tool, toolCallID string, input any, output any, isErr bool) (ToolResultOutput, error) {
	if isErr {
		if s, ok := output.(string); ok {
			return ToolResultOutput{Type: "error-text", Value: s}, nil
		}
		if output == nil {
			output = nil
		}
		return ToolResultOutput{Type: "error-json", Value: output}, nil
	}
	if tool.ToModelOutput != nil {
		return tool.ToModelOutput(toolCallID, input, output)
	}
	if s, ok := output.(string); ok {
		return ToolResultOutput{Type: "text", Value: s}, nil
	}
	if output == nil {
		return ToolResultOutput{Type: "json", Value: nil}, nil
	}
	return ToolResultOutput{Type: "json", Value: output}, nil
}
