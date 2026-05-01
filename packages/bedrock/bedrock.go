package bedrock

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/holbrookab/go-ai/internal/httputil"
	"github.com/holbrookab/go-ai/packages/ai"
	"github.com/holbrookab/go-ai/packages/bedrock/internal/sigv4"
)

type Credentials struct {
	Region          string
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

type CredentialProvider func(context.Context) (Credentials, error)

type Settings struct {
	Region             string
	APIKey             string
	AccessKeyID        string
	SecretAccessKey    string
	SessionToken       string
	BaseURL            string
	Headers            map[string]string
	Client             httputil.Doer
	CredentialProvider CredentialProvider
	GenerateID         func() string
}

type Provider struct {
	settings Settings
}

func New(settings Settings) *Provider {
	return &Provider{settings: settings}
}

func (p *Provider) LanguageModel(modelID string) ai.LanguageModel {
	return &LanguageModel{modelID: modelID, provider: p}
}

type LanguageModel struct {
	modelID  string
	provider *Provider
}

func (m *LanguageModel) Provider() string { return "amazon-bedrock" }
func (m *LanguageModel) ModelID() string  { return m.modelID }
func (m *LanguageModel) SupportedURLs(context.Context) (map[string][]string, error) {
	return map[string][]string{}, nil
}

func (m *LanguageModel) DoGenerate(ctx context.Context, opts ai.LanguageModelCallOptions) (*ai.LanguageModelGenerateResult, error) {
	body, warnings, jsonTool := m.buildBody(opts, false)
	endpoint := m.url("/converse")
	data, headers, err := m.post(ctx, endpoint, opts.Headers, body)
	if err != nil {
		return nil, err
	}
	var response converseResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, err
	}
	content := m.parseContent(response.Output.Message.Content, jsonTool)
	metadata := bedrockMetadata(response, jsonTool)
	return &ai.LanguageModelGenerateResult{
		Content:      content,
		FinishReason: ai.FinishReason{Unified: mapFinish(response.StopReason, jsonTool), Raw: response.StopReason},
		Usage:        convertUsage(response.Usage),
		Warnings:     warnings,
		ProviderMetadata: ai.ProviderMetadata{
			"amazonBedrock": metadata,
			"bedrock":       metadata,
		},
		Request: ai.RequestMetadata{Body: body},
		Response: ai.ResponseMetadata{
			Headers: headers,
			Body:    cloneRaw(response),
			ModelID: m.modelID,
		},
	}, nil
}

func (m *LanguageModel) DoStream(ctx context.Context, opts ai.LanguageModelCallOptions) (*ai.LanguageModelStreamResult, error) {
	body, warnings, jsonTool := m.buildBody(opts, true)
	endpoint := m.url("/converse-stream")
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.headers(opts.Headers, payload) {
		req.Header.Set(k, v)
	}
	if err := m.signRequest(ctx, req, payload); err != nil {
		return nil, err
	}
	resp, err := httputil.Client(m.provider.settings.Client).Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bedrock stream failed: status %d: %s", resp.StatusCode, string(data))
	}
	out := make(chan ai.StreamPart)
	go func() {
		defer resp.Body.Close()
		defer close(out)
		out <- ai.StreamPart{Type: "stream-start", Warnings: warnings}
		m.scanStream(resp.Body, jsonTool, out)
	}()
	return &ai.LanguageModelStreamResult{
		Stream:  out,
		Request: ai.RequestMetadata{Body: body},
		Response: ai.ResponseMetadata{
			Headers: responseHeaders(resp.Header),
			ModelID: m.modelID,
		},
	}, nil
}

func (m *LanguageModel) buildBody(opts ai.LanguageModelCallOptions, _ bool) (map[string]any, []ai.Warning, bool) {
	var warnings []ai.Warning
	system, messages := convertMessages(opts.Prompt)
	inference := map[string]any{}
	if opts.MaxOutputTokens != nil {
		inference["maxTokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		t := *opts.Temperature
		if t > 1 {
			warnings = append(warnings, ai.Warning{Type: "unsupported", Feature: "temperature", Details: fmt.Sprintf("%v exceeds bedrock maximum of 1.0. clamped to 1.0", t)})
			t = 1
		}
		if t < 0 {
			warnings = append(warnings, ai.Warning{Type: "unsupported", Feature: "temperature", Details: fmt.Sprintf("%v is below bedrock minimum of 0. clamped to 0", t)})
			t = 0
		}
		inference["temperature"] = t
	}
	if opts.TopP != nil {
		inference["topP"] = *opts.TopP
	}
	if opts.TopK != nil {
		inference["topK"] = *opts.TopK
	}
	if len(opts.StopSequences) > 0 {
		inference["stopSequences"] = opts.StopSequences
	}
	if opts.FrequencyPenalty != nil {
		warnings = append(warnings, ai.Warning{Type: "unsupported", Feature: "frequencyPenalty"})
	}
	if opts.PresencePenalty != nil {
		warnings = append(warnings, ai.Warning{Type: "unsupported", Feature: "presencePenalty"})
	}
	if opts.Seed != nil {
		warnings = append(warnings, ai.Warning{Type: "unsupported", Feature: "seed"})
	}

	tools := opts.Tools
	jsonTool := false
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" && opts.ResponseFormat.Schema != nil {
		jsonTool = true
		tools = append(tools, ai.ModelTool{Type: "function", Name: "json", Description: "Respond with a JSON object.", InputSchema: opts.ResponseFormat.Schema})
		opts.ToolChoice = ai.RequiredToolChoice()
	}
	body := map[string]any{
		"messages": messages,
	}
	if len(system) > 0 {
		body["system"] = system
	}
	if len(inference) > 0 {
		body["inferenceConfig"] = inference
	}
	if strings.Contains(m.modelID, "anthropic") {
		body["additionalModelResponseFieldPaths"] = []string{"/delta/stop_sequence"}
	}
	if toolConfig := bedrockToolConfig(tools, opts.ToolChoice); toolConfig != nil {
		body["toolConfig"] = toolConfig
	}
	for k, v := range bedrockOptions(opts.ProviderOptions) {
		if k == "reasoningConfig" || k == "additionalModelRequestFields" {
			continue
		}
		body[k] = v
	}
	if fields, ok := bedrockOptions(opts.ProviderOptions)["additionalModelRequestFields"]; ok {
		body["additionalModelRequestFields"] = fields
	}
	return body, warnings, jsonTool
}

func convertMessages(messages []ai.Message) ([]map[string]any, []map[string]any) {
	var system []map[string]any
	var out []map[string]any
	for _, message := range messages {
		switch message.Role {
		case ai.RoleSystem:
			system = append(system, map[string]any{"text": message.Text})
		case ai.RoleUser, ai.RoleTool:
			content := bedrockUserContent(message.Content)
			if len(content) > 0 {
				out = append(out, map[string]any{"role": "user", "content": content})
			}
		case ai.RoleAssistant:
			content := bedrockAssistantContent(message.Content)
			if len(content) > 0 {
				out = append(out, map[string]any{"role": "assistant", "content": content})
			}
		}
	}
	return system, out
}

func bedrockUserContent(parts []ai.Part) []map[string]any {
	var out []map[string]any
	for _, part := range parts {
		switch p := part.(type) {
		case ai.TextPart:
			out = append(out, map[string]any{"text": p.Text})
		case ai.FilePart:
			if strings.HasPrefix(p.MediaType, "image/") && len(p.Data.Data) > 0 {
				out = append(out, map[string]any{"image": map[string]any{"format": mediaSubtype(p.MediaType), "source": map[string]any{"bytes": base64.StdEncoding.EncodeToString(p.Data.Data)}}})
			} else if len(p.Data.Data) > 0 || p.Data.Text != "" {
				data := p.Data.Data
				if p.Data.Text != "" {
					data = []byte(p.Data.Text)
				}
				name := p.Filename
				if name == "" {
					name = "document"
				}
				out = append(out, map[string]any{"document": map[string]any{"format": mediaSubtype(p.MediaType), "name": strings.TrimSuffix(name, "."+mediaSubtype(p.MediaType)), "source": map[string]any{"bytes": base64.StdEncoding.EncodeToString(data)}}})
			}
		case ai.ToolResultPart:
			out = append(out, map[string]any{"toolResult": map[string]any{"toolUseId": p.ToolCallID, "content": []map[string]any{{"text": toolOutputText(p.Output)}}}})
		}
	}
	return out
}

func bedrockAssistantContent(parts []ai.Part) []map[string]any {
	var out []map[string]any
	for _, part := range parts {
		switch p := part.(type) {
		case ai.TextPart:
			out = append(out, map[string]any{"text": p.Text})
		case ai.ReasoningPart:
			out = append(out, map[string]any{"reasoningContent": map[string]any{"reasoningText": map[string]any{"text": p.Text}}})
		case ai.ToolCallPart:
			out = append(out, map[string]any{"toolUse": map[string]any{"toolUseId": p.ToolCallID, "name": p.ToolName, "input": p.Input}})
		}
	}
	return out
}

func bedrockToolConfig(tools []ai.ModelTool, choice ai.ToolChoice) map[string]any {
	if len(tools) == 0 || choice.Type == "none" {
		return nil
	}
	var specs []map[string]any
	for _, tool := range tools {
		if choice.Type == "tool" && choice.ToolName != "" && tool.Name != choice.ToolName {
			continue
		}
		spec := map[string]any{"name": tool.Name, "inputSchema": map[string]any{"json": tool.InputSchema}}
		if tool.Description != "" {
			spec["description"] = tool.Description
		}
		specs = append(specs, map[string]any{"toolSpec": spec})
	}
	config := map[string]any{"tools": specs}
	switch choice.Type {
	case "required":
		config["toolChoice"] = map[string]any{"any": map[string]any{}}
	case "tool":
		config["toolChoice"] = map[string]any{"tool": map[string]any{"name": choice.ToolName}}
	case "auto", "":
		config["toolChoice"] = map[string]any{"auto": map[string]any{}}
	}
	return config
}

func (m *LanguageModel) parseContent(parts []bedrockContent, jsonTool bool) []ai.Part {
	var out []ai.Part
	for _, part := range parts {
		if part.Text != nil {
			out = append(out, ai.TextPart{Text: *part.Text})
		}
		if part.ToolUse != nil {
			if jsonTool && part.ToolUse.Name == "json" {
				out = append(out, ai.TextPart{Text: mustJSON(part.ToolUse.Input)})
			} else {
				id := part.ToolUse.ToolUseID
				if id == "" {
					id = m.generateID()
				}
				out = append(out, ai.ToolCallPart{ToolCallID: id, ToolName: part.ToolUse.Name, InputRaw: mustJSON(part.ToolUse.Input)})
			}
		}
		if part.ReasoningContent != nil {
			if part.ReasoningContent.ReasoningText != nil {
				meta := ai.ProviderMetadata(nil)
				if part.ReasoningContent.ReasoningText.Signature != "" {
					meta = ai.ProviderMetadata{"amazonBedrock": map[string]any{"signature": part.ReasoningContent.ReasoningText.Signature}, "bedrock": map[string]any{"signature": part.ReasoningContent.ReasoningText.Signature}}
				}
				out = append(out, ai.ReasoningPart{Text: part.ReasoningContent.ReasoningText.Text, ProviderMetadata: meta})
			}
			if part.ReasoningContent.RedactedReasoning != nil {
				out = append(out, ai.ReasoningPart{ProviderMetadata: ai.ProviderMetadata{"amazonBedrock": map[string]any{"redactedData": part.ReasoningContent.RedactedReasoning.Data}, "bedrock": map[string]any{"redactedData": part.ReasoningContent.RedactedReasoning.Data}}})
			}
		}
	}
	return out
}

func (m *LanguageModel) scanStream(r io.Reader, jsonTool bool, out chan<- ai.StreamPart) {
	scanner := bufio.NewScanner(r)
	var usage ai.Usage
	finish := ai.FinishReason{Unified: ai.FinishOther}
	toolBuffers := map[int]*strings.Builder{}
	toolNames := map[int]string{}
	toolIDs := map[int]string{}
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
		var event map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			out <- ai.StreamPart{Type: "error", Err: err}
			continue
		}
		for typ, raw := range event {
			switch typ {
			case "contentBlockStart":
				var value struct {
					ContentBlockIndex int `json:"contentBlockIndex"`
					Start             struct {
						ToolUse *struct {
							ToolUseID string `json:"toolUseId"`
							Name      string `json:"name"`
						} `json:"toolUse"`
					} `json:"start"`
				}
				_ = json.Unmarshal(raw, &value)
				if value.Start.ToolUse != nil {
					toolBuffers[value.ContentBlockIndex] = &strings.Builder{}
					toolIDs[value.ContentBlockIndex] = value.Start.ToolUse.ToolUseID
					toolNames[value.ContentBlockIndex] = value.Start.ToolUse.Name
					out <- ai.StreamPart{Type: "tool-input-start", ToolCallID: value.Start.ToolUse.ToolUseID, ToolName: value.Start.ToolUse.Name}
				} else {
					out <- ai.StreamPart{Type: "text-start"}
				}
			case "contentBlockDelta":
				var value struct {
					ContentBlockIndex int `json:"contentBlockIndex"`
					Delta             struct {
						Text             string                  `json:"text"`
						ToolUse          *struct{ Input string } `json:"toolUse"`
						ReasoningContent *struct {
							Text string `json:"text"`
						} `json:"reasoningContent"`
					} `json:"delta"`
				}
				_ = json.Unmarshal(raw, &value)
				if value.Delta.Text != "" {
					out <- ai.StreamPart{Type: "text-delta", TextDelta: value.Delta.Text}
				}
				if value.Delta.ReasoningContent != nil {
					out <- ai.StreamPart{Type: "reasoning-delta", ReasoningDelta: value.Delta.ReasoningContent.Text}
				}
				if value.Delta.ToolUse != nil {
					if b := toolBuffers[value.ContentBlockIndex]; b != nil {
						b.WriteString(value.Delta.ToolUse.Input)
					}
					out <- ai.StreamPart{Type: "tool-input-delta", ToolInputDelta: value.Delta.ToolUse.Input, ToolCallID: toolIDs[value.ContentBlockIndex], ToolName: toolNames[value.ContentBlockIndex]}
				}
			case "contentBlockStop":
				var value struct{ ContentBlockIndex int }
				_ = json.Unmarshal(raw, &value)
				if b := toolBuffers[value.ContentBlockIndex]; b != nil {
					input := b.String()
					name := toolNames[value.ContentBlockIndex]
					id := toolIDs[value.ContentBlockIndex]
					out <- ai.StreamPart{Type: "tool-input-end", ToolCallID: id, ToolName: name, ToolInput: input}
					if jsonTool && name == "json" {
						out <- ai.StreamPart{Type: "text-delta", TextDelta: input}
					} else {
						out <- ai.StreamPart{Type: "tool-call", ToolCallID: id, ToolName: name, ToolInput: input}
					}
				} else {
					out <- ai.StreamPart{Type: "text-end"}
				}
			case "messageStop":
				var value struct {
					StopReason string `json:"stopReason"`
				}
				_ = json.Unmarshal(raw, &value)
				finish = ai.FinishReason{Unified: mapFinish(value.StopReason, jsonTool), Raw: value.StopReason}
			case "metadata":
				var value struct {
					Usage bedrockUsage `json:"usage"`
				}
				_ = json.Unmarshal(raw, &value)
				usage = convertUsage(value.Usage)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		out <- ai.StreamPart{Type: "error", Err: err}
	}
	out <- ai.StreamPart{Type: "finish", FinishReason: finish, Usage: usage}
}

func (m *LanguageModel) post(ctx context.Context, endpoint string, headers map[string]string, body any) ([]byte, map[string]string, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.headers(headers, payload) {
		req.Header.Set(k, v)
	}
	if err := m.signRequest(ctx, req, payload); err != nil {
		return nil, nil, err
	}
	resp, err := httputil.Client(m.provider.settings.Client).Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, responseHeaders(resp.Header), fmt.Errorf("bedrock request failed: status %d: %s", resp.StatusCode, string(data))
	}
	return data, responseHeaders(resp.Header), nil
}

func (m *LanguageModel) headers(headers map[string]string, body []byte) map[string]string {
	out := httputil.CloneHeaders(m.provider.settings.Headers)
	for k, v := range headers {
		out[k] = v
	}
	if out["User-Agent"] == "" {
		out["User-Agent"] = "ai-sdk/amazon-bedrock/" + ai.Version
	} else {
		out["User-Agent"] += " ai-sdk/amazon-bedrock/" + ai.Version
	}
	apiKey := strings.TrimSpace(m.provider.settings.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("AWS_BEARER_TOKEN_BEDROCK"))
	}
	if apiKey != "" {
		out["Authorization"] = "Bearer " + apiKey
		return out
	}
	return out
}

func (m *LanguageModel) url(suffix string) string {
	base := strings.TrimRight(m.provider.settings.BaseURL, "/")
	if base == "" {
		region := m.provider.settings.Region
		if region == "" {
			region = os.Getenv("AWS_REGION")
		}
		base = "https://bedrock-runtime." + region + ".amazonaws.com"
	}
	return base + "/model/" + url.PathEscape(m.modelID) + suffix
}

func (m *LanguageModel) generateID() string {
	if m.provider.settings.GenerateID != nil {
		return m.provider.settings.GenerateID()
	}
	return fmt.Sprintf("call-%d", time.Now().UnixNano())
}

type signingTransport struct {
	base  http.RoundTripper
	model *LanguageModel
}

func (m *LanguageModel) signRequest(ctx context.Context, req *http.Request, body []byte) error {
	if req.Header.Get("Authorization") != "" {
		return nil
	}
	creds, err := m.credentials(ctx)
	if err != nil {
		return err
	}
	return sigv4.Sign(req, body, sigv4.Credentials{
		Region:          creds.Region,
		AccessKeyID:     creds.AccessKeyID,
		SecretAccessKey: creds.SecretAccessKey,
		SessionToken:    creds.SessionToken,
		Service:         "bedrock",
	}, time.Now())
}

func (m *LanguageModel) credentials(ctx context.Context) (Credentials, error) {
	region := m.provider.settings.Region
	if region == "" {
		region = os.Getenv("AWS_REGION")
	}
	if m.provider.settings.CredentialProvider != nil {
		creds, err := m.provider.settings.CredentialProvider(ctx)
		if err != nil {
			return Credentials{}, err
		}
		if creds.Region == "" {
			creds.Region = region
		}
		return creds, nil
	}
	access := m.provider.settings.AccessKeyID
	secret := m.provider.settings.SecretAccessKey
	session := m.provider.settings.SessionToken
	if access == "" {
		access = os.Getenv("AWS_ACCESS_KEY_ID")
	}
	if secret == "" {
		secret = os.Getenv("AWS_SECRET_ACCESS_KEY")
	}
	if !(m.provider.settings.AccessKeyID != "" && m.provider.settings.SecretAccessKey != "") && session == "" {
		session = os.Getenv("AWS_SESSION_TOKEN")
	}
	if access == "" || secret == "" {
		return Credentials{}, fmt.Errorf("aws sigv4 authentication requires AWS credentials or AWS_BEARER_TOKEN_BEDROCK")
	}
	return Credentials{Region: region, AccessKeyID: access, SecretAccessKey: secret, SessionToken: session}, nil
}

func bedrockOptions(options ai.ProviderOptions) map[string]any {
	if options == nil {
		return nil
	}
	if v, ok := options["amazonBedrock"].(map[string]any); ok {
		return v
	}
	if v, ok := options["bedrock"].(map[string]any); ok {
		return v
	}
	return nil
}

func mediaSubtype(mediaType string) string {
	if idx := strings.Index(mediaType, "/"); idx >= 0 && idx+1 < len(mediaType) {
		return mediaType[idx+1:]
	}
	if mediaType == "" {
		return "txt"
	}
	return mediaType
}

func toolOutputText(output ai.ToolResultOutput) string {
	if output.Type == "text" || output.Type == "error-text" {
		if s, ok := output.Value.(string); ok {
			return s
		}
	}
	return mustJSON(output.Value)
}

func mapFinish(raw string, jsonTool bool) string {
	switch raw {
	case "end_turn", "stop_sequence":
		return ai.FinishStop
	case "max_tokens":
		return ai.FinishLength
	case "content_filtered", "guardrail_intervened":
		return ai.FinishContentFilter
	case "tool_use":
		if jsonTool {
			return ai.FinishStop
		}
		return ai.FinishToolCalls
	case "":
		return ai.FinishUnknown
	default:
		return ai.FinishOther
	}
}

func convertUsage(usage bedrockUsage) ai.Usage {
	inputTotal := usage.InputTokens + usage.CacheReadInputTokens + usage.CacheWriteInputTokens
	output := usage.OutputTokens
	total := inputTotal + output
	return ai.Usage{InputTokens: &inputTotal, OutputTokens: &output, TotalTokens: &total}
}

func bedrockMetadata(response converseResponse, jsonTool bool) map[string]any {
	payload := map[string]any{}
	if response.Trace != nil {
		payload["trace"] = response.Trace
	}
	if response.PerformanceConfig != nil {
		payload["performanceConfig"] = response.PerformanceConfig
	}
	if response.ServiceTier != nil {
		payload["serviceTier"] = response.ServiceTier
	}
	if jsonTool {
		payload["isJsonResponseFromTool"] = true
	}
	if stop := response.AdditionalModelResponseFields.Delta.StopSequence; stop != "" {
		payload["stopSequence"] = stop
	}
	return payload
}

func mustJSON(v any) string {
	if v == nil {
		return "{}"
	}
	b, err := json.Marshal(v)
	if err != nil {
		return "{}"
	}
	return string(b)
}

func cloneRaw(v any) any {
	b, _ := json.Marshal(v)
	var out any
	_ = json.Unmarshal(b, &out)
	return out
}

func responseHeaders(headers http.Header) map[string]string {
	out := map[string]string{}
	for k, values := range headers {
		if len(values) > 0 {
			out[k] = values[0]
		}
	}
	return out
}

type converseResponse struct {
	Output struct {
		Message struct {
			Content []bedrockContent `json:"content"`
		} `json:"message"`
	} `json:"output"`
	StopReason                    string         `json:"stopReason"`
	Usage                         bedrockUsage   `json:"usage"`
	Trace                         map[string]any `json:"trace"`
	PerformanceConfig             map[string]any `json:"performanceConfig"`
	ServiceTier                   map[string]any `json:"serviceTier"`
	AdditionalModelResponseFields struct {
		Delta struct {
			StopSequence string `json:"stop_sequence"`
		} `json:"delta"`
	} `json:"additionalModelResponseFields"`
}

type bedrockUsage struct {
	InputTokens           int `json:"inputTokens"`
	OutputTokens          int `json:"outputTokens"`
	TotalTokens           int `json:"totalTokens"`
	CacheReadInputTokens  int `json:"cacheReadInputTokens"`
	CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
}

type bedrockContent struct {
	Text    *string `json:"text"`
	ToolUse *struct {
		ToolUseID string `json:"toolUseId"`
		Name      string `json:"name"`
		Input     any    `json:"input"`
	} `json:"toolUse"`
	ReasoningContent *struct {
		ReasoningText *struct {
			Text      string `json:"text"`
			Signature string `json:"signature"`
		} `json:"reasoningText"`
		RedactedReasoning *struct {
			Data string `json:"data"`
		} `json:"redactedReasoning"`
	} `json:"reasoningContent"`
}
