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

type ToolApprovalOptions struct {
	ToolCall ToolCall
	Tools    map[string]Tool
	Messages []Message
	Context  any
}

type ToolApprovalFunction func(context.Context, ToolApprovalOptions) (ApprovalDecision, error)

type SingleToolApprovalFunction func(context.Context, ToolCall, ToolApprovalOptions) (ApprovalDecision, error)

type ToolApprovalRule struct {
	Type   string
	Reason string
	Decide SingleToolApprovalFunction
}

type ToolApprovalConfiguration struct {
	Decide ToolApprovalFunction
	Tools  map[string]ToolApprovalRule
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
	RequiresApproval bool
	NeedsApproval    func(context.Context, ToolCall) (ApprovalDecision, error)
}

const (
	ApprovalDecisionApproved      = "approved"
	ApprovalDecisionDenied        = "denied"
	ApprovalDecisionUserApproval  = "user-approval"
	ApprovalDecisionNotApplicable = "not-applicable"
)

type ApprovalDecision struct {
	Type   string
	Reason string
}

func Approved(reason ...string) ApprovalDecision {
	decision := ApprovalDecision{Type: ApprovalDecisionApproved}
	if len(reason) > 0 {
		decision.Reason = reason[0]
	}
	return decision
}

func Denied(reason ...string) ApprovalDecision {
	decision := ApprovalDecision{Type: ApprovalDecisionDenied}
	if len(reason) > 0 {
		decision.Reason = reason[0]
	}
	return decision
}

func UserApproval() ApprovalDecision {
	return ApprovalDecision{Type: ApprovalDecisionUserApproval}
}

func RequireUserApproval() func(context.Context, ToolCall) (ApprovalDecision, error) {
	return func(context.Context, ToolCall) (ApprovalDecision, error) {
		return UserApproval(), nil
	}
}

func resolveToolApproval(ctx context.Context, tools map[string]Tool, call ToolCall, approval *ToolApprovalConfiguration, messages []Message, toolsContext map[string]any) (ApprovalDecision, error) {
	options := ToolApprovalOptions{
		ToolCall: call,
		Tools:    tools,
		Messages: messages,
		Context:  toolsContext,
	}
	if approval != nil {
		if approval.Decide != nil {
			decision, err := approval.Decide(ctx, options)
			if err != nil {
				return ApprovalDecision{}, err
			}
			return normalizeApprovalDecision(decision), nil
		}
		if rule, ok := approval.Tools[call.ToolName]; ok {
			if rule.Decide != nil {
				decision, err := rule.Decide(ctx, call, options)
				if err != nil {
					return ApprovalDecision{}, err
				}
				return normalizeApprovalDecision(decision), nil
			}
			return normalizeApprovalDecision(ApprovalDecision{Type: rule.Type, Reason: rule.Reason}), nil
		}
	}

	tool, ok := tools[call.ToolName]
	if !ok {
		return ApprovalDecision{Type: ApprovalDecisionNotApplicable}, nil
	}
	if tool.NeedsApproval != nil {
		decision, err := tool.NeedsApproval(ctx, call)
		if err != nil {
			return ApprovalDecision{}, err
		}
		return normalizeApprovalDecision(decision), nil
	}
	if tool.RequiresApproval {
		return UserApproval(), nil
	}
	return ApprovalDecision{Type: ApprovalDecisionNotApplicable}, nil
}

func normalizeApprovalDecision(decision ApprovalDecision) ApprovalDecision {
	if decision.Type == "" {
		decision.Type = ApprovalDecisionNotApplicable
	}
	return decision
}

func ApprovalBlocksToolExecution(decision ApprovalDecision) bool {
	return decision.Type == ApprovalDecisionDenied || decision.Type == ApprovalDecisionUserApproval
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

func ValidateToolInput(tool Tool, input any) error {
	if tool.InputSchema != nil {
		if err := validateJSONSchema(normalizeSchema(tool.InputSchema), input, "$"); err != nil {
			return err
		}
	}
	if tool.ValidateInput != nil {
		if err := tool.ValidateInput(input); err != nil {
			return err
		}
	}
	return nil
}

func ValidateToolOutput(tool Tool, output any) error {
	if tool.OutputSchema != nil {
		if err := validateJSONSchema(normalizeSchema(tool.OutputSchema), output, "$"); err != nil {
			return err
		}
	}
	if tool.ValidateOutput != nil {
		if err := tool.ValidateOutput(output); err != nil {
			return err
		}
	}
	return nil
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
