package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/holbrookab/go-ai/packages/ai"
)

func TestAPIKeyExpressModeAndRequestMapping(t *testing.T) {
	client := &captureClient{response: `{
		"candidates":[{"content":{"parts":[{"text":"hello"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3}
	}`}
	provider := New(Settings{
		APIKey:     "key",
		Client:     client,
		GenerateID: func() string { return "id" },
	})
	model := provider.LanguageModel("gemini-2.5-flash")
	result, err := model.DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.SystemMessage("system"), ai.UserMessage("hello")},
		ResponseFormat: &ai.ResponseFormat{
			Type:   "json",
			Schema: map[string]any{"type": "object"},
		},
		ProviderOptions: ai.ProviderOptions{
			"googleVertex": map[string]any{
				"serviceTier":    "flex",
				"safetySettings": []any{map[string]any{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}},
			},
		},
		Tools: []ai.ModelTool{{
			Type:        "function",
			Name:        "weather",
			Description: "Weather",
			InputSchema: map[string]any{"type": "object"},
		}},
		ToolChoice: ai.ToolChoiceFor("weather"),
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if got := client.request.Header.Get("x-goog-api-key"); got != "key" {
		t.Fatalf("expected api key header, got %q", got)
	}
	if client.request.Header.Get("Authorization") != "" {
		t.Fatalf("did not expect authorization header in express mode")
	}
	if got := client.request.URL.String(); got != "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent" {
		t.Fatalf("unexpected url: %s", got)
	}
	var body map[string]any
	if err := json.Unmarshal(client.body, &body); err != nil {
		t.Fatal(err)
	}
	if body["serviceTier"] != "SERVICE_TIER_FLEX" {
		t.Fatalf("expected mapped service tier, got %#v", body["serviceTier"])
	}
	gen := body["generationConfig"].(map[string]any)
	if gen["responseMimeType"] != "application/json" {
		t.Fatalf("expected json response mime type, got %#v", gen)
	}
	if _, ok := body["systemInstruction"]; !ok {
		t.Fatalf("expected systemInstruction in body: %#v", body)
	}
	if ai.TextFromParts(result.Content) != "hello" {
		t.Fatalf("expected parsed text, got %q", ai.TextFromParts(result.Content))
	}
}

func TestOAuthGlobalBaseURL(t *testing.T) {
	client := &captureClient{response: `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`}
	provider := New(Settings{
		Project:     "proj",
		Location:    "global",
		Client:      client,
		TokenSource: aiToken("tok"),
	})
	_, err := provider.LanguageModel("gemini-2.5-flash").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	if got := client.request.Header.Get("Authorization"); got != "Bearer tok" {
		t.Fatalf("expected oauth header, got %q", got)
	}
	if got := client.request.URL.String(); got != "https://aiplatform.googleapis.com/v1beta1/projects/proj/locations/global/publishers/google/models/gemini-2.5-flash:generateContent" {
		t.Fatalf("unexpected url: %s", got)
	}
}

func TestMessageTextFallbackAndEmptyContentSkipped(t *testing.T) {
	client := &captureClient{response: `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`}
	provider := New(Settings{
		APIKey: "key",
		Client: client,
	})
	_, err := provider.LanguageModel("gemini-2.5-flash").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{
			{Role: ai.RoleUser, Text: "hello from temporal wire"},
			{Role: ai.RoleAssistant},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}
	var body struct {
		Contents []struct {
			Role  string           `json:"role"`
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(client.body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Contents) != 1 {
		t.Fatalf("expected empty assistant content to be skipped, got %#v", body.Contents)
	}
	if body.Contents[0].Role != "user" || len(body.Contents[0].Parts) != 1 || body.Contents[0].Parts[0]["text"] != "hello from temporal wire" {
		t.Fatalf("unexpected contents: %#v", body.Contents)
	}
}

func TestThoughtSignatureRoundTripsInGenerateContent(t *testing.T) {
	client := &captureClient{response: `{
		"candidates":[{
			"content":{"parts":[
				{"text":"visible","thoughtSignature":"text-sig"},
				{"text":"reason","thought":true,"thoughtSignature":"reason-sig"},
				{"functionCall":{"name":"extractDocument","args":{"id":"doc-1"}},"thoughtSignature":"tool-sig"}
			]},
			"finishReason":"STOP"
		}]
	}`}
	provider := New(Settings{
		APIKey:     "key",
		Client:     client,
		GenerateID: sequenceID("call-1"),
	})
	result, err := provider.LanguageModel("gemini-2.5-flash").DoGenerate(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{
			{
				Role: ai.RoleAssistant,
				Content: []ai.Part{
					ai.TextPart{Text: "visible", ProviderMetadata: ai.ProviderMetadata{"google": map[string]any{"thoughtSignature": "text-sig"}}},
					ai.ReasoningPart{Text: "reason", ProviderMetadata: ai.ProviderMetadata{"vertex": map[string]any{"thoughtSignature": "reason-sig"}}},
					ai.ToolCallPart{ToolCallID: "call-1", ToolName: "extractDocument", Input: map[string]any{"id": "doc-1"}, ProviderMetadata: ai.ProviderMetadata{"googleVertex": map[string]any{"thoughtSignature": "tool-sig"}}},
				},
			},
			{
				Role: ai.RoleTool,
				Content: []ai.Part{
					ai.ToolResultPart{ToolCallID: "call-1", ToolName: "extractDocument", Output: ai.ToolResultOutput{Type: "text", Value: "done"}, ProviderMetadata: ai.ProviderMetadata{"googleVertex": map[string]any{"thoughtSignature": "tool-sig"}}},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("DoGenerate failed: %v", err)
	}

	var body struct {
		Contents []struct {
			Parts []map[string]any `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(client.body, &body); err != nil {
		t.Fatal(err)
	}
	if got := body.Contents[0].Parts[0]["thoughtSignature"]; got != "text-sig" {
		t.Fatalf("text thought signature = %#v", got)
	}
	if got := body.Contents[0].Parts[1]["thoughtSignature"]; got != "reason-sig" {
		t.Fatalf("reasoning thought signature = %#v", got)
	}
	if got := body.Contents[0].Parts[2]["thoughtSignature"]; got != "tool-sig" {
		t.Fatalf("tool-call thought signature = %#v", got)
	}
	if _, ok := body.Contents[1].Parts[0]["thoughtSignature"]; ok {
		t.Fatalf("functionResponse should not include thoughtSignature: %#v", body.Contents[1].Parts[0])
	}

	text, ok := result.Content[0].(ai.TextPart)
	if !ok || thoughtSignature(text.ProviderMetadata, nil) != "text-sig" {
		t.Fatalf("text part metadata not preserved: %#v", result.Content[0])
	}
	reasoning, ok := result.Content[1].(ai.ReasoningPart)
	if !ok || thoughtSignature(reasoning.ProviderMetadata, nil) != "reason-sig" {
		t.Fatalf("reasoning metadata not preserved: %#v", result.Content[1])
	}
	call, ok := result.Content[2].(ai.ToolCallPart)
	if !ok || thoughtSignature(call.ProviderMetadata, nil) != "tool-sig" {
		t.Fatalf("tool call metadata not preserved: %#v", result.Content[2])
	}
}

func TestStreamParsesSSE(t *testing.T) {
	client := &captureClient{response: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"he\"}]}}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"llo\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":1,\"candidatesTokenCount\":2}}\n\n"}
	provider := New(Settings{
		APIKey:     "key",
		Client:     client,
		GenerateID: func() string { return "id" },
	})
	result, err := provider.LanguageModel("gemini-2.5-flash").DoStream(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}
	var text string
	var finish ai.StreamPart
	for part := range result.Stream {
		if part.Type == "text-delta" {
			text += part.TextDelta
		}
		if part.Type == "finish" {
			finish = part
		}
	}
	if text != "hello" {
		t.Fatalf("expected streamed hello, got %q", text)
	}
	if finish.FinishReason.Unified != ai.FinishStop {
		t.Fatalf("expected stop finish, got %#v", finish.FinishReason)
	}
}

func TestStreamToolCallPreservesThoughtSignature(t *testing.T) {
	client := &captureClient{response: "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"extractDocument\",\"args\":{\"id\":\"doc-1\"}},\"thoughtSignature\":\"stream-sig\"}]},\"finishReason\":\"STOP\"}]}\n\n"}
	provider := New(Settings{
		APIKey:     "key",
		Client:     client,
		GenerateID: sequenceID("call-1"),
	})
	result, err := provider.LanguageModel("gemini-2.5-flash").DoStream(context.Background(), ai.LanguageModelCallOptions{
		Prompt: []ai.Message{ai.UserMessage("hello")},
	})
	if err != nil {
		t.Fatalf("DoStream failed: %v", err)
	}
	var call ai.StreamPart
	for part := range result.Stream {
		if part.Type == "tool-call" {
			call = part
		}
	}
	if call.ToolName != "extractDocument" || thoughtSignature(call.ProviderMetadata, nil) != "stream-sig" {
		t.Fatalf("stream tool call metadata not preserved: %#v", call)
	}
}

type aiToken string

func (t aiToken) Token(context.Context) (string, error) { return string(t), nil }

func sequenceID(ids ...string) func() string {
	index := 0
	return func() string {
		if index >= len(ids) {
			return "id"
		}
		id := ids[index]
		index++
		return id
	}
}

type captureClient struct {
	request  *http.Request
	body     []byte
	response string
}

func (c *captureClient) Do(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	req.Body = io.NopCloser(bytes.NewReader(body))
	c.request = req
	c.body = body
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(c.response)),
	}, nil
}
