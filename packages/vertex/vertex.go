package vertex

import (
	"bufio"
	"bytes"
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/holbrookab/go-ai/internal/httputil"
	"github.com/holbrookab/go-ai/packages/ai"
)

const (
	expressBaseURL = "https://aiplatform.googleapis.com/v1/publishers/google"
	oauthScope     = "https://www.googleapis.com/auth/cloud-platform"
)

type TokenSource interface {
	Token(context.Context) (string, error)
}

type TokenSourceFunc func(context.Context) (string, error)

func (f TokenSourceFunc) Token(ctx context.Context) (string, error) { return f(ctx) }

type Settings struct {
	APIKey      string
	Location    string
	Project     string
	BaseURL     string
	Headers     map[string]string
	Client      httputil.Doer
	TokenSource TokenSource
	GenerateID  func() string
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

func (m *LanguageModel) Provider() string { return "google.vertex.chat" }
func (m *LanguageModel) ModelID() string  { return m.modelID }
func (m *LanguageModel) SupportedURLs(context.Context) (map[string][]string, error) {
	return map[string][]string{"*": {"^https?://.*$", "^gs://.*$"}}, nil
}

func (m *LanguageModel) DoGenerate(ctx context.Context, opts ai.LanguageModelCallOptions) (*ai.LanguageModelGenerateResult, error) {
	body, warnings := m.buildBody(opts, false)
	headers, err := m.headers(ctx, opts.Headers)
	if err != nil {
		return nil, err
	}
	data, responseHeaders, err := httputil.PostJSON(ctx, m.provider.settings.Client, m.url(":generateContent"), headers, body)
	if err != nil {
		return nil, err
	}
	var response generateContentResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, err
	}
	content := m.parseContent(response)
	finishRaw := ""
	var metadata ai.ProviderMetadata
	var usage ai.Usage
	if len(response.Candidates) > 0 {
		finishRaw = response.Candidates[0].FinishReason
		metadata = m.metadata(response)
		usage = convertUsage(response.UsageMetadata)
	}
	return &ai.LanguageModelGenerateResult{
		Content:          content,
		FinishReason:     ai.FinishReason{Unified: mapFinish(finishRaw, hasClientToolCall(content)), Raw: finishRaw},
		Usage:            usage,
		Warnings:         warnings,
		ProviderMetadata: metadata,
		Request:          ai.RequestMetadata{Body: body},
		Response:         ai.ResponseMetadata{Headers: responseHeaders, Body: cloneRaw(response), ModelID: m.modelID},
	}, nil
}

func (m *LanguageModel) DoStream(ctx context.Context, opts ai.LanguageModelCallOptions) (*ai.LanguageModelStreamResult, error) {
	body, warnings := m.buildBody(opts, true)
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	headers, err := m.headers(ctx, opts.Headers)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.url(":streamGenerateContent?alt=sse"), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := httputil.Client(m.provider.settings.Client).Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("vertex stream failed: status %d: %s", resp.StatusCode, string(data))
	}
	out := make(chan ai.StreamPart)
	go func() {
		defer resp.Body.Close()
		defer close(out)
		out <- ai.StreamPart{Type: "stream-start", Warnings: warnings}
		m.scanSSE(resp.Body, out)
	}()
	return &ai.LanguageModelStreamResult{
		Stream:   out,
		Request:  ai.RequestMetadata{Body: body},
		Response: ai.ResponseMetadata{Headers: responseHeaders(resp.Header), ModelID: m.modelID},
	}, nil
}

func (m *LanguageModel) buildBody(opts ai.LanguageModelCallOptions, streaming bool) (map[string]any, []ai.Warning) {
	contents, systemInstruction := googleMessages(opts.Prompt, strings.HasPrefix(strings.ToLower(m.modelID), "gemma-"))
	generation := map[string]any{}
	if opts.MaxOutputTokens != nil {
		generation["maxOutputTokens"] = *opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		generation["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		generation["topP"] = *opts.TopP
	}
	if opts.TopK != nil {
		generation["topK"] = *opts.TopK
	}
	if opts.FrequencyPenalty != nil {
		generation["frequencyPenalty"] = *opts.FrequencyPenalty
	}
	if opts.PresencePenalty != nil {
		generation["presencePenalty"] = *opts.PresencePenalty
	}
	if len(opts.StopSequences) > 0 {
		generation["stopSequences"] = opts.StopSequences
	}
	if opts.Seed != nil {
		generation["seed"] = *opts.Seed
	}
	if opts.ResponseFormat != nil && opts.ResponseFormat.Type == "json" {
		generation["responseMimeType"] = "application/json"
		if opts.ResponseFormat.Schema != nil {
			generation["responseSchema"] = opts.ResponseFormat.Schema
		}
	}
	vertexOptions := providerOptions(opts.ProviderOptions)
	for _, key := range []string{"responseModalities", "thinkingConfig", "mediaResolution", "imageConfig", "audioTimestamp"} {
		if v, ok := vertexOptions[key]; ok {
			generation[key] = v
		}
	}
	body := map[string]any{"contents": contents}
	if len(systemInstruction) > 0 {
		body["systemInstruction"] = map[string]any{"parts": systemInstruction}
	}
	if len(generation) > 0 {
		body["generationConfig"] = generation
	}
	for _, key := range []string{"safetySettings", "cachedContent", "labels"} {
		if v, ok := vertexOptions[key]; ok {
			body[key] = v
		}
	}
	if tier, ok := vertexOptions["serviceTier"].(string); ok {
		body["serviceTier"] = mapServiceTier(tier)
	}
	if len(opts.Tools) > 0 {
		body["tools"] = googleTools(opts.Tools)
		if cfg := googleToolConfig(opts.Tools, opts.ToolChoice, streaming, vertexOptions); cfg != nil {
			body["toolConfig"] = cfg
		}
	}
	return body, nil
}

func googleMessages(messages []ai.Message, isGemma bool) ([]map[string]any, []map[string]string) {
	var system []map[string]string
	var contents []map[string]any
	var pendingSystem string
	for _, message := range messages {
		switch message.Role {
		case ai.RoleSystem:
			if isGemma {
				if pendingSystem != "" {
					pendingSystem += "\n"
				}
				pendingSystem += message.Text
			} else {
				system = append(system, map[string]string{"text": message.Text})
			}
		case ai.RoleUser:
			parts := googleUserParts(message.Content)
			if pendingSystem != "" {
				parts = append([]map[string]any{{"text": pendingSystem}}, parts...)
				pendingSystem = ""
			}
			contents = append(contents, map[string]any{"role": "user", "parts": parts})
		case ai.RoleAssistant:
			contents = append(contents, map[string]any{"role": "model", "parts": googleAssistantParts(message.Content)})
		case ai.RoleTool:
			contents = append(contents, map[string]any{"role": "user", "parts": googleToolResultParts(message.Content)})
		}
	}
	if pendingSystem != "" {
		contents = append([]map[string]any{{"role": "user", "parts": []map[string]any{{"text": pendingSystem}}}}, contents...)
	}
	return contents, system
}

func googleUserParts(parts []ai.Part) []map[string]any {
	var out []map[string]any
	for _, part := range parts {
		switch p := part.(type) {
		case ai.TextPart:
			out = append(out, map[string]any{"text": p.Text})
		case ai.FilePart:
			if p.Data.URL != "" {
				out = append(out, map[string]any{"fileData": map[string]any{"mimeType": p.MediaType, "fileUri": p.Data.URL}})
			} else if p.Data.Text != "" {
				out = append(out, map[string]any{"inlineData": map[string]any{"mimeType": mediaType(p.MediaType, "text/plain"), "data": base64.StdEncoding.EncodeToString([]byte(p.Data.Text))}})
			} else if len(p.Data.Data) > 0 {
				out = append(out, map[string]any{"inlineData": map[string]any{"mimeType": mediaType(p.MediaType, "application/octet-stream"), "data": base64.StdEncoding.EncodeToString(p.Data.Data)}})
			}
		}
	}
	return out
}

func googleAssistantParts(parts []ai.Part) []map[string]any {
	var out []map[string]any
	for _, part := range parts {
		switch p := part.(type) {
		case ai.TextPart:
			out = append(out, map[string]any{"text": p.Text})
		case ai.ReasoningPart:
			out = append(out, map[string]any{"text": p.Text, "thought": true})
		case ai.ToolCallPart:
			out = append(out, map[string]any{"functionCall": map[string]any{"name": p.ToolName, "args": p.Input}})
		}
	}
	return out
}

func googleToolResultParts(parts []ai.Part) []map[string]any {
	var out []map[string]any
	for _, part := range parts {
		if p, ok := part.(ai.ToolResultPart); ok {
			out = append(out, map[string]any{"functionResponse": map[string]any{"name": p.ToolName, "response": map[string]any{"name": p.ToolName, "content": toolOutputText(p.Output)}}})
		}
	}
	return out
}

func googleTools(tools []ai.ModelTool) []map[string]any {
	var declarations []map[string]any
	var providerTools []map[string]any
	for _, tool := range tools {
		if tool.Type == "provider" {
			providerTools = append(providerTools, providerTool(tool))
			continue
		}
		declaration := map[string]any{"name": tool.Name, "parameters": tool.InputSchema}
		if tool.Description != "" {
			declaration["description"] = tool.Description
		}
		declarations = append(declarations, declaration)
	}
	var out []map[string]any
	if len(declarations) > 0 {
		out = append(out, map[string]any{"functionDeclarations": declarations})
	}
	out = append(out, providerTools...)
	return out
}

func providerTool(tool ai.ModelTool) map[string]any {
	switch tool.ID {
	case "google.google_search":
		return map[string]any{"googleSearch": tool.Args}
	case "google.url_context":
		return map[string]any{"urlContext": tool.Args}
	case "google.code_execution":
		return map[string]any{"codeExecution": tool.Args}
	case "google.vertex_rag_store":
		return map[string]any{"retrieval": tool.Args}
	case "google.google_maps":
		return map[string]any{"googleMaps": tool.Args}
	case "google.enterprise_web_search":
		return map[string]any{"enterpriseWebSearch": tool.Args}
	default:
		return map[string]any{tool.ID: tool.Args}
	}
}

func googleToolConfig(tools []ai.ModelTool, choice ai.ToolChoice, streaming bool, opts map[string]any) map[string]any {
	if choice.Type == "none" {
		return map[string]any{"functionCallingConfig": map[string]any{"mode": "NONE"}}
	}
	mode := "AUTO"
	cfg := map[string]any{}
	switch choice.Type {
	case "required":
		mode = "ANY"
	case "tool":
		mode = "ANY"
		cfg["allowedFunctionNames"] = []string{choice.ToolName}
	}
	cfg["mode"] = mode
	if streaming {
		if v, ok := opts["streamFunctionCallArguments"].(bool); ok && v {
			cfg["streamFunctionCallArguments"] = true
		}
	}
	return map[string]any{"functionCallingConfig": cfg}
}

func (m *LanguageModel) parseContent(response generateContentResponse) []ai.Part {
	if len(response.Candidates) == 0 {
		return nil
	}
	var out []ai.Part
	var lastCodeID string
	for _, part := range response.Candidates[0].Content.Parts {
		switch {
		case part.Text != "":
			if part.Thought {
				out = append(out, ai.ReasoningPart{Text: part.Text, ProviderMetadata: thoughtMetadata(part.ThoughtSignature)})
			} else {
				out = append(out, ai.TextPart{Text: part.Text, ProviderOptions: nil})
			}
		case part.FunctionCall != nil:
			out = append(out, ai.ToolCallPart{ToolCallID: m.generateID(), ToolName: part.FunctionCall.Name, InputRaw: mustJSON(part.FunctionCall.Args), ProviderMetadata: thoughtMetadata(part.ThoughtSignature)})
		case part.InlineData != nil:
			file := ai.FilePart{MediaType: part.InlineData.MimeType, Data: ai.FileData{Type: "data", Data: []byte(part.InlineData.Data)}, ProviderOptions: nil}
			if part.Thought {
				out = append(out, ai.ReasoningFilePart{MediaType: file.MediaType, Data: file.Data, ProviderMetadata: thoughtMetadata(part.ThoughtSignature)})
			} else {
				out = append(out, file)
			}
		case part.ExecutableCode != nil:
			lastCodeID = m.generateID()
			out = append(out, ai.ToolCallPart{ToolCallID: lastCodeID, ToolName: "code_execution", InputRaw: mustJSON(part.ExecutableCode), ProviderExecuted: true})
		case part.CodeExecutionResult != nil:
			out = append(out, ai.ToolResultPart{ToolCallID: lastCodeID, ToolName: "code_execution", Result: part.CodeExecutionResult, ProviderExecuted: true})
		}
	}
	for _, source := range extractSources(response.Candidates[0].GroundingMetadata, m.generateID) {
		out = append(out, source)
	}
	return out
}

func (m *LanguageModel) scanSSE(r io.Reader, out chan<- ai.StreamPart) {
	scanner := bufio.NewScanner(r)
	finish := ai.FinishReason{Unified: ai.FinishOther}
	var usage ai.Usage
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if line == "[DONE]" {
			break
		}
		var chunk generateContentResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			out <- ai.StreamPart{Type: "error", Err: err}
			continue
		}
		if chunk.UsageMetadata.PromptTokenCount != 0 || chunk.UsageMetadata.CandidatesTokenCount != 0 {
			usage = convertUsage(chunk.UsageMetadata)
		}
		if len(chunk.Candidates) == 0 {
			continue
		}
		c := chunk.Candidates[0]
		if c.FinishReason != "" {
			finish = ai.FinishReason{Unified: mapFinish(c.FinishReason, false), Raw: c.FinishReason}
		}
		for _, part := range c.Content.Parts {
			if part.Text != "" {
				if part.Thought {
					out <- ai.StreamPart{Type: "reasoning-delta", ReasoningDelta: part.Text}
				} else {
					out <- ai.StreamPart{Type: "text-delta", TextDelta: part.Text}
				}
			}
			if part.FunctionCall != nil && part.FunctionCall.Name != "" {
				input := mustJSON(part.FunctionCall.Args)
				id := m.generateID()
				out <- ai.StreamPart{Type: "tool-input-start", ToolCallID: id, ToolName: part.FunctionCall.Name}
				out <- ai.StreamPart{Type: "tool-input-delta", ToolCallID: id, ToolName: part.FunctionCall.Name, ToolInputDelta: input}
				out <- ai.StreamPart{Type: "tool-input-end", ToolCallID: id, ToolName: part.FunctionCall.Name, ToolInput: input}
				out <- ai.StreamPart{Type: "tool-call", ToolCallID: id, ToolName: part.FunctionCall.Name, ToolInput: input}
			}
			if part.InlineData != nil {
				out <- ai.StreamPart{Type: "file", Content: ai.FilePart{MediaType: part.InlineData.MimeType, Data: ai.FileData{Type: "data", Data: []byte(part.InlineData.Data)}}}
			}
		}
		for _, source := range extractSources(c.GroundingMetadata, m.generateID) {
			out <- ai.StreamPart{Type: "source", Content: source}
		}
	}
	if err := scanner.Err(); err != nil {
		out <- ai.StreamPart{Type: "error", Err: err}
	}
	out <- ai.StreamPart{Type: "finish", FinishReason: finish, Usage: usage}
}

func (m *LanguageModel) url(suffix string) string {
	base := strings.TrimRight(m.baseURL(), "/")
	path := m.modelID
	if !strings.Contains(path, "/") {
		path = "models/" + path
	}
	return base + "/" + path + suffix
}

func (m *LanguageModel) baseURL() string {
	settings := m.provider.settings
	if settings.BaseURL != "" {
		return settings.BaseURL
	}
	if m.apiKey() != "" {
		return expressBaseURL
	}
	location := settings.Location
	if location == "" {
		location = os.Getenv("GOOGLE_VERTEX_LOCATION")
	}
	project := settings.Project
	if project == "" {
		project = os.Getenv("GOOGLE_VERTEX_PROJECT")
	}
	host := "aiplatform.googleapis.com"
	if location != "global" {
		host = location + "-aiplatform.googleapis.com"
	}
	return fmt.Sprintf("https://%s/v1beta1/projects/%s/locations/%s/publishers/google", host, project, location)
}

func (m *LanguageModel) headers(ctx context.Context, headers map[string]string) (map[string]string, error) {
	out := httputil.CloneHeaders(m.provider.settings.Headers)
	for k, v := range headers {
		out[k] = v
	}
	if out["User-Agent"] == "" {
		out["User-Agent"] = "ai-sdk/google-vertex/" + ai.Version
	} else {
		out["User-Agent"] += " ai-sdk/google-vertex/" + ai.Version
	}
	if apiKey := m.apiKey(); apiKey != "" {
		out["x-goog-api-key"] = apiKey
		return out, nil
	}
	source := m.provider.settings.TokenSource
	if source == nil {
		source = defaultTokenSource{client: m.provider.settings.Client}
	}
	token, err := source.Token(ctx)
	if err != nil {
		return nil, err
	}
	out["Authorization"] = "Bearer " + token
	return out, nil
}

func (m *LanguageModel) apiKey() string {
	if strings.TrimSpace(m.provider.settings.APIKey) != "" {
		return strings.TrimSpace(m.provider.settings.APIKey)
	}
	return strings.TrimSpace(os.Getenv("GOOGLE_VERTEX_API_KEY"))
}

func (m *LanguageModel) generateID() string {
	if m.provider.settings.GenerateID != nil {
		return m.provider.settings.GenerateID()
	}
	return fmt.Sprintf("call-%d", time.Now().UnixNano())
}

func providerOptions(options ai.ProviderOptions) map[string]any {
	for _, key := range []string{"googleVertex", "vertex", "google"} {
		if v, ok := options[key].(map[string]any); ok {
			return v
		}
	}
	return map[string]any{}
}

func mapServiceTier(tier string) string {
	switch tier {
	case "standard":
		return "SERVICE_TIER_STANDARD"
	case "flex":
		return "SERVICE_TIER_FLEX"
	case "priority":
		return "SERVICE_TIER_PRIORITY"
	default:
		return tier
	}
}

func mapFinish(raw string, hasTools bool) string {
	switch raw {
	case "STOP":
		if hasTools {
			return ai.FinishToolCalls
		}
		return ai.FinishStop
	case "MAX_TOKENS":
		return ai.FinishLength
	case "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "IMAGE_SAFETY":
		return ai.FinishContentFilter
	case "MALFORMED_FUNCTION_CALL":
		return ai.FinishError
	case "":
		return ai.FinishUnknown
	default:
		return ai.FinishOther
	}
}

func convertUsage(u usageMetadata) ai.Usage {
	cached := u.CachedContentTokenCount
	input := u.PromptTokenCount
	noCache := input - cached
	if noCache < 0 {
		noCache = input
	}
	reasoning := u.ThoughtsTokenCount
	output := u.CandidatesTokenCount + reasoning
	total := input + output
	return ai.Usage{InputTokens: &noCache, OutputTokens: &output, TotalTokens: &total, ReasoningTokens: &reasoning, CachedInputTokens: &cached}
}

func hasClientToolCall(parts []ai.Part) bool {
	for _, part := range parts {
		if call, ok := part.(ai.ToolCallPart); ok && !call.ProviderExecuted {
			return true
		}
	}
	return false
}

func thoughtMetadata(signature string) ai.ProviderMetadata {
	if signature == "" {
		return nil
	}
	payload := map[string]any{"thoughtSignature": signature}
	return ai.ProviderMetadata{"googleVertex": payload, "vertex": payload}
}

func (m *LanguageModel) metadata(response generateContentResponse) ai.ProviderMetadata {
	payload := map[string]any{"usageMetadata": response.UsageMetadata, "promptFeedback": response.PromptFeedback}
	if len(response.Candidates) > 0 {
		payload["groundingMetadata"] = response.Candidates[0].GroundingMetadata
		payload["urlContextMetadata"] = response.Candidates[0].URLContextMetadata
		payload["safetyRatings"] = response.Candidates[0].SafetyRatings
		payload["finishMessage"] = response.Candidates[0].FinishMessage
	}
	if response.ServiceTier != "" {
		payload["serviceTier"] = response.ServiceTier
	}
	return ai.ProviderMetadata{"googleVertex": payload, "vertex": payload}
}

func extractSources(metadata groundingMetadata, generateID func() string) []ai.SourcePart {
	var out []ai.SourcePart
	seen := map[string]bool{}
	for _, chunk := range metadata.GroundingChunks {
		if chunk.Web.URI != "" && !seen[chunk.Web.URI] {
			seen[chunk.Web.URI] = true
			out = append(out, ai.SourcePart{ID: generateID(), SourceType: "url", URL: chunk.Web.URI, Title: chunk.Web.Title})
		}
		if chunk.RetrievedContext.URI != "" && !seen[chunk.RetrievedContext.URI] {
			seen[chunk.RetrievedContext.URI] = true
			out = append(out, ai.SourcePart{ID: generateID(), SourceType: "url", URL: chunk.RetrievedContext.URI, Title: chunk.RetrievedContext.Title})
		}
	}
	return out
}

func mediaType(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func toolOutputText(output ai.ToolResultOutput) string {
	if output.Type == "text" || output.Type == "error-text" {
		if s, ok := output.Value.(string); ok {
			return s
		}
	}
	return mustJSON(output.Value)
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

type generateContentResponse struct {
	Candidates     []candidate   `json:"candidates"`
	UsageMetadata  usageMetadata `json:"usageMetadata"`
	PromptFeedback any           `json:"promptFeedback"`
	ServiceTier    string        `json:"serviceTier"`
}

type candidate struct {
	Content            googleContent     `json:"content"`
	FinishReason       string            `json:"finishReason"`
	FinishMessage      string            `json:"finishMessage"`
	GroundingMetadata  groundingMetadata `json:"groundingMetadata"`
	URLContextMetadata any               `json:"urlContextMetadata"`
	SafetyRatings      any               `json:"safetyRatings"`
}

type googleContent struct {
	Role  string       `json:"role"`
	Parts []googlePart `json:"parts"`
}

type googlePart struct {
	Text                string         `json:"text"`
	Thought             bool           `json:"thought"`
	ThoughtSignature    string         `json:"thoughtSignature"`
	FunctionCall        *functionCall  `json:"functionCall"`
	InlineData          *inlineData    `json:"inlineData"`
	ExecutableCode      map[string]any `json:"executableCode"`
	CodeExecutionResult map[string]any `json:"codeExecutionResult"`
}

type functionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

type groundingMetadata struct {
	GroundingChunks []struct {
		Web struct {
			URI   string `json:"uri"`
			Title string `json:"title"`
		} `json:"web"`
		RetrievedContext struct {
			URI   string `json:"uri"`
			Title string `json:"title"`
		} `json:"retrievedContext"`
	} `json:"groundingChunks"`
}

type defaultTokenSource struct {
	client httputil.Doer
}

func (s defaultTokenSource) Token(ctx context.Context) (string, error) {
	if token := os.Getenv("GOOGLE_VERTEX_ACCESS_TOKEN"); token != "" {
		return token, nil
	}
	if path := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"); path != "" {
		return serviceAccountToken(ctx, s.client, path)
	}
	return metadataToken(ctx, s.client)
}

func metadataToken(ctx context.Context, client httputil.Doer) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token?scopes="+oauthScope, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	resp, err := httputil.Client(client).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("metadata token request failed: %s", string(data))
	}
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return "", err
	}
	if payload.AccessToken == "" {
		return "", fmt.Errorf("metadata token response did not include access_token")
	}
	return payload.AccessToken, nil
}

func serviceAccountToken(ctx context.Context, client httputil.Doer, path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var key struct {
		ClientEmail string `json:"client_email"`
		PrivateKey  string `json:"private_key"`
		TokenURI    string `json:"token_uri"`
	}
	if err := json.Unmarshal(data, &key); err != nil {
		return "", err
	}
	block, _ := pem.Decode([]byte(key.PrivateKey))
	if block == nil {
		return "", fmt.Errorf("service account private key is not PEM")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return "", err
	}
	privateKey, ok := parsed.(*rsa.PrivateKey)
	if !ok {
		return "", fmt.Errorf("service account private key is not RSA")
	}
	now := time.Now()
	claims := map[string]any{"iss": key.ClientEmail, "scope": oauthScope, "aud": key.TokenURI, "iat": now.Unix(), "exp": now.Add(time.Hour).Unix()}
	assertion, err := signedJWT(claims, privateKey)
	if err != nil {
		return "", err
	}
	form := "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion=" + assertion
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, key.TokenURI, strings.NewReader(form))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := httputil.Client(client).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("service account token request failed: %s", string(body))
	}
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", err
	}
	return payload.AccessToken, nil
}

func signedJWT(claims map[string]any, key *rsa.PrivateKey) (string, error) {
	header := map[string]any{"alg": "RS256", "typ": "JWT"}
	h, _ := json.Marshal(header)
	c, _ := json.Marshal(claims)
	signingInput := base64.RawURLEncoding.EncodeToString(h) + "." + base64.RawURLEncoding.EncodeToString(c)
	digest := sha256.Sum256([]byte(signingInput))
	sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, digest[:])
	if err != nil {
		return "", err
	}
	return signingInput + "." + base64.RawURLEncoding.EncodeToString(sig), nil
}
