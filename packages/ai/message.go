package ai

import "encoding/json"

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role            Role
	Content         []Part
	Text            string
	ProviderOptions ProviderOptions
}

func SystemMessage(text string) Message {
	return Message{Role: RoleSystem, Text: text}
}

func UserMessage(text string) Message {
	return Message{Role: RoleUser, Content: []Part{TextPart{Text: text}}}
}

func AssistantMessage(text string) Message {
	return Message{Role: RoleAssistant, Content: []Part{TextPart{Text: text}}}
}

func ToolMessage(parts ...ToolResultPart) Message {
	content := make([]Part, len(parts))
	for i := range parts {
		content[i] = parts[i]
	}
	return Message{Role: RoleTool, Content: content}
}

type Part interface {
	part()
	PartType() string
}

type TextPart struct {
	Text            string
	ProviderOptions ProviderOptions
}

func (TextPart) part()            {}
func (TextPart) PartType() string { return "text" }

type FileData struct {
	Type              string
	Data              []byte
	Text              string
	URL               string
	Reference         string
	ProviderReference ProviderReference
}

type FilePart struct {
	Data            FileData
	MediaType       string
	Filename        string
	ProviderOptions ProviderOptions
}

func (FilePart) part()            {}
func (FilePart) PartType() string { return "file" }

type ReasoningPart struct {
	Text             string
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ReasoningPart) part()            {}
func (ReasoningPart) PartType() string { return "reasoning" }

type ReasoningFilePart struct {
	Data             FileData
	MediaType        string
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ReasoningFilePart) part()            {}
func (ReasoningFilePart) PartType() string { return "reasoning-file" }

type ToolCallPart struct {
	ToolCallID       string
	ToolName         string
	Input            any
	InputRaw         string
	ProviderExecuted bool
	Dynamic          bool
	Invalid          bool
	Error            error
	Title            string
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ToolCallPart) part()            {}
func (ToolCallPart) PartType() string { return "tool-call" }

func (p ToolCallPart) InputJSON() string {
	if p.InputRaw != "" {
		return p.InputRaw
	}
	b, err := json.Marshal(p.Input)
	if err != nil {
		return "{}"
	}
	return string(b)
}

type ToolResultOutput struct {
	Type            string
	Value           any
	Reason          string
	ProviderOptions ProviderOptions
}

type ToolResultPart struct {
	ToolCallID       string
	ToolName         string
	Input            any
	Output           ToolResultOutput
	Result           any
	IsError          bool
	ProviderExecuted bool
	Dynamic          bool
	Preliminary      bool
	ProviderMetadata ProviderMetadata
	ProviderOptions  ProviderOptions
}

func (ToolResultPart) part()            {}
func (ToolResultPart) PartType() string { return "tool-result" }

type SourcePart struct {
	ID         string
	SourceType string
	URL        string
	Title      string
}

func (SourcePart) part()            {}
func (SourcePart) PartType() string { return "source" }

func TextFromParts(parts []Part) string {
	var out string
	for _, part := range parts {
		if text, ok := part.(TextPart); ok {
			out += text.Text
		}
	}
	return out
}
