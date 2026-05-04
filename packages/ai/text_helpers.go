package ai

import "context"

type PruneReasoningMode string

const (
	PruneReasoningNone              PruneReasoningMode = "none"
	PruneReasoningAll               PruneReasoningMode = "all"
	PruneReasoningBeforeLastMessage PruneReasoningMode = "before-last-message"
)

type PruneEmptyMessagesMode string

const (
	PruneEmptyMessagesRemove PruneEmptyMessagesMode = "remove"
	PruneEmptyMessagesKeep   PruneEmptyMessagesMode = "keep"
)

type PruneToolCallsMode string

const (
	PruneToolCallsAllMode               PruneToolCallsMode = "all"
	PruneToolCallsBeforeLastMessageMode PruneToolCallsMode = "before-last-message"
)

type PruneToolCallsRule struct {
	Mode             PruneToolCallsMode
	BeforeLastN      int
	Tools            []string
	KeepLastMessages int
}

func PruneAllToolCalls(tools ...string) PruneToolCallsRule {
	return PruneToolCallsRule{Mode: PruneToolCallsAllMode, Tools: tools}
}

func PruneToolCallsBeforeLastMessage(tools ...string) PruneToolCallsRule {
	return PruneToolCallsRule{Mode: PruneToolCallsBeforeLastMessageMode, Tools: tools}
}

func PruneToolCallsBeforeLastMessages(n int, tools ...string) PruneToolCallsRule {
	return PruneToolCallsRule{BeforeLastN: n, Tools: tools}
}

type PruneMessagesOptions struct {
	Messages      []Message
	Reasoning     PruneReasoningMode
	ToolCalls     []PruneToolCallsRule
	EmptyMessages PruneEmptyMessagesMode
}

func FilterActiveTools(tools map[string]Tool, activeTools []string) map[string]Tool {
	if tools == nil || activeTools == nil {
		return tools
	}

	active := make(map[string]struct{}, len(activeTools))
	for _, name := range activeTools {
		active[name] = struct{}{}
	}

	filtered := make(map[string]Tool, len(active))
	for name, tool := range tools {
		if _, ok := active[name]; ok {
			filtered[name] = tool
		}
	}
	return filtered
}

func ExtractTextContent(content []Part) (string, bool) {
	var text string
	found := false
	for _, part := range content {
		if textPart, ok := part.(TextPart); ok {
			text += textPart.Text
			found = true
		}
	}
	return text, found
}

func ExtractReasoningContent(content []Part) (string, bool) {
	var text string
	found := false
	for _, part := range content {
		reasoningPart, ok := part.(ReasoningPart)
		if !ok {
			continue
		}
		if found {
			text += "\n"
		}
		text += reasoningPart.Text
		found = true
	}
	return text, found
}

func PruneMessages(opts PruneMessagesOptions) []Message {
	messages := cloneMessages(opts.Messages)
	reasoning := opts.Reasoning
	if reasoning == "" {
		reasoning = PruneReasoningNone
	}
	emptyMessages := opts.EmptyMessages
	if emptyMessages == "" {
		emptyMessages = PruneEmptyMessagesRemove
	}

	if reasoning == PruneReasoningAll || reasoning == PruneReasoningBeforeLastMessage {
		for i := range messages {
			if messages[i].Role != RoleAssistant {
				continue
			}
			if reasoning == PruneReasoningBeforeLastMessage && i == len(messages)-1 {
				continue
			}
			messages[i].Content = filterParts(messages[i].Content, func(part Part) bool {
				_, isReasoning := part.(ReasoningPart)
				return !isReasoning
			})
		}
	}

	for _, rule := range opts.ToolCalls {
		keepLastMessages := rule.keepLastCount()
		keptToolCallIDs := map[string]struct{}{}

		if keepLastMessages != nil {
			start := len(messages) - *keepLastMessages
			if start < 0 {
				start = 0
			}
			for _, message := range messages[start:] {
				if message.Role != RoleAssistant && message.Role != RoleTool {
					continue
				}
				for _, part := range message.Content {
					if id := toolPartID(part); id != "" {
						keptToolCallIDs[id] = struct{}{}
					}
				}
			}
		}

		for i := range messages {
			if messages[i].Role != RoleAssistant && messages[i].Role != RoleTool {
				continue
			}
			if keepLastMessages != nil && i >= len(messages)-*keepLastMessages {
				continue
			}
			messages[i].Content = filterParts(messages[i].Content, func(part Part) bool {
				id := toolPartID(part)
				if id == "" {
					return true
				}
				if _, keep := keptToolCallIDs[id]; keep {
					return true
				}
				return !rule.matchesTool(toolPartName(part))
			})
		}
	}

	if emptyMessages == PruneEmptyMessagesRemove {
		messages = filterMessages(messages, func(message Message) bool {
			return message.Text != "" || len(message.Content) > 0
		})
	}

	return messages
}

func ResolveToolApproval(ctx context.Context, tools map[string]Tool, call ToolCall) (ApprovalDecision, error) {
	return ResolveToolApprovalWithConfiguration(ctx, tools, call, nil, nil, nil)
}

func ResolveToolApprovalWithConfiguration(ctx context.Context, tools map[string]Tool, call ToolCall, approval *ToolApprovalConfiguration, messages []Message, toolsContext map[string]any) (ApprovalDecision, error) {
	return resolveToolApproval(ctx, tools, call, approval, messages, toolsContext)
}

func (r PruneToolCallsRule) keepLastCount() *int {
	switch {
	case r.KeepLastMessages > 0:
		return &r.KeepLastMessages
	case r.Mode == PruneToolCallsBeforeLastMessageMode:
		n := 1
		return &n
	case r.BeforeLastN > 0:
		return &r.BeforeLastN
	default:
		return nil
	}
}

func (r PruneToolCallsRule) matchesTool(toolName string) bool {
	if r.Tools == nil {
		return true
	}
	for _, tool := range r.Tools {
		if tool == toolName {
			return true
		}
	}
	return false
}

func cloneMessages(messages []Message) []Message {
	out := make([]Message, len(messages))
	for i, message := range messages {
		out[i] = message
		if message.Content != nil {
			out[i].Content = append([]Part{}, message.Content...)
		}
	}
	return out
}

func filterParts(parts []Part, keep func(Part) bool) []Part {
	out := parts[:0]
	for _, part := range parts {
		if keep(part) {
			out = append(out, part)
		}
	}
	return out
}

func filterMessages(messages []Message, keep func(Message) bool) []Message {
	out := messages[:0]
	for _, message := range messages {
		if keep(message) {
			out = append(out, message)
		}
	}
	return out
}

func toolPartID(part Part) string {
	switch p := part.(type) {
	case ToolCallPart:
		return p.ToolCallID
	case ToolResultPart:
		return p.ToolCallID
	default:
		return ""
	}
}

func toolPartName(part Part) string {
	switch p := part.(type) {
	case ToolCallPart:
		return p.ToolName
	case ToolResultPart:
		return p.ToolName
	default:
		return ""
	}
}
